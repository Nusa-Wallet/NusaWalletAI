"""Combine per-model backtest predictions into one weighted ensemble forecast.

Joins the models on (pair, origin_date, target_date, split, horizon), takes a per-pair
weighted average of point and quantile forecasts, and records model disagreement (the
standard deviation of the model point forecasts) — the confidence signal for the engine.
"""

import numpy as np
import pandas as pd

KEY = ["pair", "origin_date", "target_date", "evaluation_split", "horizon"]
_FORECAST_COLS = ["point_forecast", "q10", "q50", "q90"]


def combine_predictions(
    per_model: dict[str, pd.DataFrame],
    weights: dict[str, dict[str, float]],
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Return an ensemble prediction frame with disagreement, one row per case."""
    available = [m for m in models if m in per_model]
    if not available:
        raise ValueError("No requested models present in predictions")
    stacked = pd.concat([per_model[m].assign(model=m) for m in available], ignore_index=True)

    rows: list[dict] = []
    for key_values, group in stacked.groupby(KEY, sort=False):
        pair = key_values[0]
        group_models = group["model"].tolist()
        w = weights.get(pair, {})
        raw = np.array([w.get(m, 0.0) for m in group_models], dtype=float)
        if raw.sum() <= 0:
            raw = np.ones(len(group_models))
        weight_vec = raw / raw.sum()

        record = dict(zip(KEY, key_values))
        record["current_rate"] = float(group["current_rate"].iloc[0])
        record["actual_rate"] = float(group["actual_rate"].iloc[0])
        for col in _FORECAST_COLS:
            record[col] = float(np.sum(weight_vec * group[col].to_numpy(dtype=float)))
        record["disagreement"] = float(np.std(group["point_forecast"].to_numpy(dtype=float)))
        record["n_models"] = int(len(group_models))
        rows.append(record)

    return pd.DataFrame.from_records(rows)
