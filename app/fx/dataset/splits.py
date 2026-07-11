"""Chronological split labelling and walk-forward window generation.

The split boundaries are global date quantiles applied identically to every pair, so
train precedes val precedes test in time with no leakage. Walk-forward windows (for the
Phase 9 backtest) are expressed over the sorted unique observation dates.
"""

import pandas as pd

from app.fx.dataset.config import FxDatasetConfig


def assign_split(panel: pd.DataFrame, config: FxDatasetConfig) -> pd.DataFrame:
    dates = pd.Series(sorted(panel["date"].unique()))
    train_cut = dates.quantile(config.train_frac)
    val_cut = dates.quantile(config.train_frac + config.val_frac)
    split = pd.Series("test", index=panel.index, dtype="object")
    split[panel["date"] <= train_cut] = "train"
    split[(panel["date"] > train_cut) & (panel["date"] <= val_cut)] = "val"
    panel = panel.copy()
    panel["split"] = split
    return panel


def make_walk_forward_windows(panel: pd.DataFrame, config: FxDatasetConfig) -> list[dict]:
    """Rolling-origin windows over unique observation dates (business days)."""
    dates = sorted(pd.to_datetime(panel["date"].unique()))
    n = len(dates)
    init, horizon, step = config.wf_initial_train_obs, config.wf_horizon_obs, config.wf_step_obs
    windows: list[dict] = []
    origin = init
    while origin + horizon <= n:
        windows.append({
            "train_start": dates[0].date().isoformat(),
            "train_end": dates[origin - 1].date().isoformat(),
            "test_start": dates[origin].date().isoformat(),
            "test_end": dates[origin + horizon - 1].date().isoformat(),
            "horizon_obs": horizon,
        })
        origin += step
    return windows
