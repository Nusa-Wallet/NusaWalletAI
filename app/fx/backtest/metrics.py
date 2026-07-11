"""Forecast and fee-aware conversion metrics for Phase 9."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def compute_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot compute metrics for an empty backtest frame")
    actual = frame["actual_rate"].to_numpy(dtype=float)
    predicted = frame["point_forecast"].to_numpy(dtype=float)
    current = frame["current_rate"].to_numpy(dtype=float)
    q10 = frame["q10"].to_numpy(dtype=float)
    q50 = frame["q50"].to_numpy(dtype=float)
    q90 = frame["q90"].to_numpy(dtype=float)
    error = actual - predicted
    direction_actual = np.sign(actual - current)
    direction_predicted = np.sign(predicted - current)
    gains = frame["net_gain_vs_immediate"].to_numpy(dtype=float)
    regrets = frame["regret"].to_numpy(dtype=float)
    immediate = frame["immediate_net"].to_numpy(dtype=float)
    chosen = frame["chosen_net"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float(np.mean(np.abs(error / actual))),
        "directional_accuracy": float(np.mean(direction_actual == direction_predicted)),
        "pinball_q10": pinball_loss(actual, q10, 0.1),
        "pinball_q50": pinball_loss(actual, q50, 0.5),
        "pinball_q90": pinball_loss(actual, q90, 0.9),
        "mean_pinball": float(np.mean([
            pinball_loss(actual, q10, 0.1),
            pinball_loss(actual, q50, 0.5),
            pinball_loss(actual, q90, 0.9),
        ])),
        "interval_coverage_80": float(np.mean((actual >= q10) & (actual <= q90))),
        "mean_interval_width": float(np.mean(q90 - q10)),
        "wait_rate": float(np.mean(frame["action"].eq("WAIT"))),
        "decision_win_rate": float(np.mean(chosen >= immediate)),
        "mean_net_gain_vs_immediate": float(np.mean(gains)),
        "total_net_gain_vs_immediate": float(np.sum(gains)),
        "mean_regret": float(np.mean(regrets)),
        "maximum_regret": float(np.max(regrets)),
    }


def grouped_metrics(predictions: pd.DataFrame) -> dict[str, dict]:
    output: dict[str, dict] = {}
    columns = ["model", "pair", "evaluation_split", "horizon"]
    for keys, group in predictions.groupby(columns, sort=True):
        model, pair, split, horizon = keys
        key = f"{model}|{pair}|{split}|h{int(horizon)}"
        output[key] = compute_metrics(group)
    return output
