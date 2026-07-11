"""CatBoost hyperparameter tuning.

Uses Optuna when installed (Kaggle); otherwise falls back to a deterministic random
search so the pipeline is fully runnable locally. Objective: maximise validation
average precision (PR-AUC), which is the right target for imbalanced fraud data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from app.fraud.training.models import default_catboost_params, train_catboost

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def _params_from(seed: int, depth: int, lr: float, l2: float, iterations: int) -> dict:
    params = default_catboost_params(seed)
    params.update({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2, "iterations": iterations})
    return params


def _score(params, x_tr, y_tr, x_val, y_val, seed) -> float:
    model = train_catboost(x_tr, y_tr, x_val, y_val, params=params, seed=seed)
    return float(average_precision_score(y_val, model.predict_proba(x_val)[:, 1]))


def tune_catboost(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 25,
    seed: int = 42,
) -> tuple[dict, float, str]:
    """Return (best_params, best_val_pr_auc, method)."""
    if HAS_OPTUNA:
        def objective(trial):
            params = _params_from(
                seed,
                depth=trial.suggest_int("depth", 4, 8),
                lr=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                l2=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                iterations=trial.suggest_int("iterations", 200, 600, step=100),
            )
            return _score(params, x_tr, y_tr, x_val, y_val, seed)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = _params_from(
            seed,
            study.best_params["depth"],
            study.best_params["learning_rate"],
            study.best_params["l2_leaf_reg"],
            study.best_params["iterations"],
        )
        return best, float(study.best_value), "optuna"

    # Deterministic random search fallback.
    rng = np.random.default_rng(seed)
    best_params, best_val = default_catboost_params(seed), -1.0
    for _ in range(n_trials):
        params = _params_from(
            seed,
            depth=int(rng.integers(4, 9)),
            lr=float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
            l2=float(rng.uniform(1.0, 10.0)),
            iterations=int(rng.choice([200, 300, 400, 500, 600])),
        )
        val = _score(params, x_tr, y_tr, x_val, y_val, seed)
        if val > best_val:
            best_params, best_val = params, val
    return best_params, best_val, "random_search"
