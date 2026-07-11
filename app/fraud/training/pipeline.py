"""End-to-end fraud training orchestration.

Trains the five plan experiments (rules-only, isolation-forest, catboost-v1,
catboost-tuned, full-ensemble), tunes on validation, calibrates the ensemble, selects
a threshold under the FPR ceiling, evaluates once on the held-out test split, logs to
MLflow (best-effort), and writes reloadable artifacts with full provenance metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import catboost

from app.fraud.explain.shap_explain import global_shap_summary
from app.fraud.feature_spec import MODEL_FEATURES
from app.fraud.simulation.config import DATASET_VERSION, SCHEMA_VERSION
from app.fraud.training.data import feature_matrix, load_dataset, time_split, to_xy
from app.fraud.training.ensemble import (
    DEFAULT_WEIGHTS,
    EnsembleModel,
    blend,
    fit_calibrator,
    select_threshold,
)
from app.fraud.training.evaluate import compute_metrics, recall_per_anomaly_type
from app.fraud.training.models import (
    catboost_proba,
    default_catboost_params,
    rules_score,
    train_catboost,
    train_isolation,
)
from app.fraud.training.tuning import tune_catboost

_REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_VERSION = "fraud-ensemble-1.0.0"

CATBOOST_FILE = "fraud_catboost.cbm"
ISOLATION_FILE = "fraud_isolation.joblib"
ENSEMBLE_FILE = "fraud_calibrator.joblib"  # holds EnsembleModel (calibrator+weights+threshold)
METADATA_FILE = "fraud_metadata.json"
SHAP_SUMMARY_FILE = "fraud_shap_summary.json"


@dataclass
class TrainingConfig:
    dataset_path: str = str(_REPO_ROOT / "data" / "synthetic" / f"{DATASET_VERSION}.parquet")
    artifacts_dir: str = str(_REPO_ROOT / "artifacts")
    train_frac: float = 0.6
    val_frac: float = 0.2
    n_trials: int = 25
    seed: int = 42
    max_fpr: float = 0.2
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    mlflow_enabled: bool = True
    mlflow_experiment: str = "fraud"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def predict_risk(catboost_model, isolation, ensemble: EnsembleModel, df: pd.DataFrame) -> np.ndarray:
    """Compute the calibrated ensemble risk for a frame (shared by serving/tests)."""
    x = feature_matrix(df)
    supervised = catboost_proba(catboost_model, x)
    anomaly = isolation.score(x)
    rules = rules_score(df)
    return ensemble.risk(supervised, anomaly, rules)


def save_bundle(artifacts_dir: Path, catboost_model, isolation, ensemble, metadata: dict) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    catboost_model.save_model(str(artifacts_dir / CATBOOST_FILE))
    joblib.dump(isolation, artifacts_dir / ISOLATION_FILE)
    joblib.dump(ensemble, artifacts_dir / ENSEMBLE_FILE)
    (artifacts_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_bundle(artifacts_dir: Path):
    model = catboost.CatBoostClassifier()
    model.load_model(str(artifacts_dir / CATBOOST_FILE))
    isolation = joblib.load(artifacts_dir / ISOLATION_FILE)
    ensemble = joblib.load(artifacts_dir / ENSEMBLE_FILE)
    metadata = json.loads((artifacts_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return model, isolation, ensemble, metadata


class _MlflowLogger:
    """Best-effort MLflow logging; never breaks training if MLflow is unavailable."""

    def __init__(self, enabled: bool, experiment: str):
        self.mlflow = None
        if not enabled:
            return
        try:
            import os

            import mlflow

            # MLflow 3.x deprecated the file store; use the repo sqlite backend
            # (matching the existing mlflow.db) unless the user overrides the URI.
            uri = os.environ.get(
                "MLFLOW_TRACKING_URI", f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}"
            )
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
        except Exception:
            self.mlflow = None

    def log(self, name: str, params: dict, metrics: dict) -> None:
        if self.mlflow is None:
            return
        try:
            with self.mlflow.start_run(run_name=name):
                self.mlflow.log_params(params)
                self.mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception:
            pass


def run_training(config: TrainingConfig | None = None) -> dict:
    config = config or TrainingConfig()
    logger = _MlflowLogger(config.mlflow_enabled, config.mlflow_experiment)

    df = load_dataset(config.dataset_path)
    splits = time_split(df, config.train_frac, config.val_frac)
    x_tr, y_tr = to_xy(splits.train)
    x_val, y_val = to_xy(splits.val)
    x_te, y_te = to_xy(splits.test)
    y_val_a, y_te_a = y_val.to_numpy(), y_te.to_numpy()

    # --- component scores per split ---
    base_params = default_catboost_params(config.seed)
    cb_v1 = train_catboost(x_tr, y_tr, x_val, y_val, params=base_params, seed=config.seed)
    sup_v1 = {"val": catboost_proba(cb_v1, x_val), "test": catboost_proba(cb_v1, x_te)}

    best_params, best_val_prauc, tune_method = tune_catboost(
        x_tr, y_tr, x_val, y_val, n_trials=config.n_trials, seed=config.seed
    )
    cb_tuned = train_catboost(x_tr, y_tr, x_val, y_val, params=best_params, seed=config.seed)
    sup = {"val": catboost_proba(cb_tuned, x_val), "test": catboost_proba(cb_tuned, x_te)}

    # Global SHAP importance (Phase 6) on a test sample for the model card.
    shap_summary = global_shap_summary(cb_tuned, x_te.head(min(2000, len(x_te))))

    contamination = float(np.clip(y_tr.mean(), 0.01, 0.3))
    isolation = train_isolation(x_tr, contamination=contamination, seed=config.seed)
    anom = {"val": isolation.score(x_val), "test": isolation.score(x_te)}

    rules = {"val": rules_score(splits.val), "test": rules_score(splits.test)}

    # --- ensemble: blend on val, calibrate, choose threshold ---
    raw_val = blend(sup["val"], anom["val"], rules["val"], config.weights)
    raw_test = blend(sup["test"], anom["test"], rules["test"], config.weights)
    calibrator = fit_calibrator(raw_val, y_val_a)
    ensemble = EnsembleModel(weights=dict(config.weights), calibrator=calibrator, threshold=0.5)
    ens_val = ensemble.risk(sup["val"], anom["val"], rules["val"])
    ens_test = ensemble.risk(sup["test"], anom["test"], rules["test"])
    ensemble.threshold = select_threshold(y_val_a, ens_val, max_fpr=config.max_fpr)

    # --- evaluate each experiment on the untouched test split ---
    def evaluate(risk_val, risk_test):
        t = select_threshold(y_val_a, risk_val, max_fpr=config.max_fpr)
        return compute_metrics(y_te_a, risk_test, t)

    experiments = {
        "rules-only": evaluate(rules["val"], rules["test"]),
        "isolation-forest": evaluate(anom["val"], anom["test"]),
        "catboost-v1": evaluate(sup_v1["val"], sup_v1["test"]),
        "catboost-tuned": evaluate(sup["val"], sup["test"]),
        "full-ensemble": compute_metrics(y_te_a, ens_test, ensemble.threshold),
    }
    flagged_test = ens_test >= ensemble.threshold
    per_type = {
        "full-ensemble": recall_per_anomaly_type(splits.test, flagged_test),
        "rules-only": recall_per_anomaly_type(
            splits.test, rules["test"] >= experiments["rules-only"]["threshold"]
        ),
    }

    for name, metrics in experiments.items():
        params = best_params if name == "catboost-tuned" else base_params if "catboost" in name else {}
        logger.log(name, {"experiment": name, **{k: params.get(k) for k in ("depth", "learning_rate", "iterations")}}, metrics)

    # --- definition-of-done checks ---
    ens_m, rules_m = experiments["full-ensemble"], experiments["rules-only"]
    dod = {
        "fpr_below_target": bool(ens_m["fpr"] < config.max_fpr),
        "beats_rules_f1": bool(ens_m["f1"] >= rules_m["f1"]),
        "beats_rules_pr_auc": bool((ens_m["pr_auc"] or 0) >= (rules_m["pr_auc"] or 0)),
        "calibrated_brier": ens_m["brier"],
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": config.seed,
        "feature_names": list(MODEL_FEATURES),
        "category_mappings": {},  # all features numeric
        "ensemble_weights": ensemble.weights,
        "threshold": ensemble.threshold,
        "tuning": {"method": tune_method, "n_trials": config.n_trials, "best_val_pr_auc": round(best_val_prauc, 4)},
        "catboost_params": best_params,
        "split": {
            "train_frac": config.train_frac, "val_frac": config.val_frac,
            "rows": {"train": len(x_tr), "val": len(x_val), "test": len(x_te)},
        },
        "metrics": experiments,
        "recall_per_anomaly_type": per_type,
        "shap_global_summary_top": dict(list(shap_summary.items())[:10]),
        "definition_of_done": dod,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": catboost.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    artifacts_dir = Path(config.artifacts_dir)
    save_bundle(artifacts_dir, cb_tuned, isolation, ensemble, metadata)
    (artifacts_dir / SHAP_SUMMARY_FILE).write_text(json.dumps(shap_summary, indent=2), encoding="utf-8")

    return {"experiments": experiments, "definition_of_done": dod, "metadata": metadata,
            "recall_per_anomaly_type": per_type, "shap_global_summary": shap_summary,
            "artifacts_dir": str(artifacts_dir)}
