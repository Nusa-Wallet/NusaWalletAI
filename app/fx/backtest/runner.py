"""Batched rolling-origin backtest for zero-shot FX models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess
import uuid

import numpy as np
import pandas as pd

from app.fx.backtest.contracts import ForecastModel
from app.fx.backtest.metrics import grouped_metrics
from app.fx.dataset.config import DATASET_VERSION, PRIMARY_PAIRS, SCHEMA_VERSION

BACKTEST_VERSION = "fx-backtest-1.0.0"


@dataclass(frozen=True)
class BacktestConfig:
    pairs: tuple[str, ...] = PRIMARY_PAIRS
    horizons: tuple[int, ...] = (1, 3, 7)
    context_length: int = 1024
    batch_size: int = 32
    fee_rate: float = 0.005
    amount_base: float = 1000.0
    evaluation_splits: tuple[str, ...] = ("val", "test")
    max_windows: int | None = None

    def validate(self) -> "BacktestConfig":
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("Horizons must be positive")
        if self.context_length < 2 or self.batch_size < 1:
            raise ValueError("Context length and batch size must be positive")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("Fee rate must be in [0, 1)")
        return self


@dataclass(frozen=True)
class _Case:
    pair: str
    origin_date: pd.Timestamp
    evaluation_split: str
    context: np.ndarray
    future_rates: np.ndarray
    future_dates: tuple[pd.Timestamp, ...]


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _load_windows(metadata: dict, max_windows: int | None) -> list[dict]:
    windows = list(metadata["walk_forward"]["windows"])
    return windows[-max_windows:] if max_windows else windows


def _build_cases(panel: pd.DataFrame, metadata: dict, config: BacktestConfig) -> list[_Case]:
    max_horizon = max(config.horizons)
    windows = _load_windows(metadata, config.max_windows)
    cases: list[_Case] = []
    for pair in config.pairs:
        series = panel[panel["pair"] == pair].sort_values("date").reset_index(drop=True)
        if series.empty:
            raise ValueError(f"Pair is missing from FX dataset: {pair}")
        for window in windows:
            train_end = pd.Timestamp(window["train_end"], tz="UTC")
            test_start = pd.Timestamp(window["test_start"], tz="UTC")
            history = series[series["date"] <= train_end]
            future = series[series["date"] >= test_start].head(max_horizon)
            if len(history) < 2 or len(future) < max_horizon:
                continue
            if future["split"].nunique() != 1:
                # Never let one forecast case straddle validation and test.
                continue
            split = str(future.iloc[0]["split"])
            if split not in config.evaluation_splits:
                continue
            context = history["rate"].tail(config.context_length).to_numpy(dtype=float)
            cases.append(_Case(
                pair=pair,
                origin_date=pd.Timestamp(history.iloc[-1]["date"]),
                evaluation_split=split,
                context=context,
                future_rates=future["rate"].to_numpy(dtype=float),
                future_dates=tuple(pd.Timestamp(value) for value in future["date"]),
            ))
    if not cases:
        raise ValueError("No eligible walk-forward cases for the selected pairs/splits")
    return cases


def _records_for_batch(
    model: ForecastModel,
    cases: list[_Case],
    forecast,
    config: BacktestConfig,
) -> list[dict]:
    records: list[dict] = []
    keep = 1 - config.fee_rate
    for row_index, case in enumerate(cases):
        current = float(case.context[-1])
        for horizon in config.horizons:
            step = horizon - 1
            actual = float(case.future_rates[step])
            point = float(forecast.point[row_index, step])
            q10 = float(forecast.quantiles[0.1][row_index, step])
            q50 = float(forecast.quantiles[0.5][row_index, step])
            q90 = float(forecast.quantiles[0.9][row_index, step])
            expected_return = point / current - 1
            action = "WAIT" if expected_return > config.fee_rate else "CONVERT_NOW"
            immediate_net = config.amount_base * current * keep
            wait_net = config.amount_base * actual * keep
            chosen_net = wait_net if action == "WAIT" else immediate_net
            oracle_rate = max(current, float(np.max(case.future_rates[:horizon])))
            oracle_net = config.amount_base * oracle_rate * keep
            records.append({
                "model": model.name,
                "model_version": model.version,
                "pair": case.pair,
                "origin_date": case.origin_date,
                "target_date": case.future_dates[step],
                "evaluation_split": case.evaluation_split,
                "horizon": horizon,
                "current_rate": current,
                "actual_rate": actual,
                "point_forecast": point,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "action": action,
                "immediate_net": immediate_net,
                "chosen_net": chosen_net,
                "oracle_net": oracle_net,
                "net_gain_vs_immediate": chosen_net - immediate_net,
                "regret": oracle_net - chosen_net,
            })
    return records


def run_backtest(
    panel: pd.DataFrame,
    dataset_metadata: dict,
    models: list[ForecastModel],
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    config = (config or BacktestConfig()).validate()
    if not models:
        raise ValueError("At least one forecast model is required")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    cases = _build_cases(panel, dataset_metadata, config)
    max_horizon = max(config.horizons)
    records: list[dict] = []
    for model in models:
        for start in range(0, len(cases), config.batch_size):
            batch = cases[start:start + config.batch_size]
            forecast = model.forecast_batch([case.context for case in batch], max_horizon)
            forecast.validate(len(batch), max_horizon)
            records.extend(_records_for_batch(model, batch, forecast, config))
    predictions = pd.DataFrame.from_records(records)
    metrics = grouped_metrics(predictions)
    metadata = {
        "backtest_version": BACKTEST_VERSION,
        "dataset_version": dataset_metadata.get("dataset_version", DATASET_VERSION),
        "schema_version": dataset_metadata.get("schema_version", SCHEMA_VERSION),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
        "models": [{"name": model.name, "version": model.version} for model in models],
        "config": {**asdict(config), "pairs": list(config.pairs), "horizons": list(config.horizons),
                   "evaluation_splits": list(config.evaluation_splits)},
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "metrics": metrics,
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__},
    }
    return predictions, metadata


def save_backtest(
    predictions: pd.DataFrame,
    metadata: dict,
    output_root: Path,
    run_id: str | None = None,
) -> Path:
    directory = output_root / (run_id or uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    (directory / "metrics.json").write_text(
        json.dumps(metadata["metrics"], indent=2), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory


def log_mlflow(metadata: dict, artifact_dir: Path, experiment: str = "fx-zero-shot") -> None:
    """Best-effort logging; local artifacts remain authoritative if MLflow fails."""
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        for model in metadata["models"]:
            with mlflow.start_run(run_name=model["name"]):
                mlflow.log_params({
                    "model_version": model["version"],
                    "backtest_version": metadata["backtest_version"],
                    "dataset_version": metadata["dataset_version"],
                    "horizons": ",".join(map(str, metadata["config"]["horizons"])),
                    "pairs": ",".join(metadata["config"]["pairs"]),
                })
                prefix = model["name"] + "|"
                for key, values in metadata["metrics"].items():
                    if not key.startswith(prefix):
                        continue
                    metric_prefix = key.replace("|", ".").replace("/", "_")
                    for metric, value in values.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{metric_prefix}.{metric}", value)
                mlflow.log_artifacts(str(artifact_dir))
    except Exception:
        return
