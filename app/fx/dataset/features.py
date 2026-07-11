"""Feature engineering for the FX panel.

All features are computed per pair, in date order, and are known "as of" the close of
day t (log-return at t, lags of prior returns, trailing rolling stats). Nothing uses a
future observation, so they are safe inputs for forecasting horizon >= 1. Warm-up rows
(before enough history exists) carry NaN by design — no aggressive interpolation.
"""

import numpy as np
import pandas as pd

from app.fx.dataset.config import FxDatasetConfig


def add_features(panel: pd.DataFrame, config: FxDatasetConfig) -> pd.DataFrame:
    """Add log-return, return lags, trailing rolling stats, calendar and gap columns."""
    panel = panel.sort_values(["pair", "date"], kind="mergesort").reset_index(drop=True)
    grouped = panel.groupby("pair", sort=False)

    log_rate = np.log(panel["rate"])
    panel["log_return"] = log_rate.groupby(panel["pair"], sort=False).diff()

    ret = panel.groupby("pair", sort=False)["log_return"]
    for lag in config.return_lags:
        panel[f"return_lag_{lag}"] = ret.shift(lag)
    for window in config.rolling_windows:
        # Trailing window ending at t (t's realised return is known at t).
        roll = ret.rolling(window, min_periods=window)
        panel[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        panel[f"roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)

    panel["day_of_week"] = panel["date"].dt.dayofweek.astype("int64")
    # Calendar gap to the previous observation (documents weekends/holidays).
    gap = grouped["date"].diff().dt.days
    panel["gap_days"] = gap
    return panel


def feature_columns(config: FxDatasetConfig) -> list[str]:
    cols = ["log_return"]
    cols += [f"return_lag_{lag}" for lag in config.return_lags]
    for window in config.rolling_windows:
        cols += [f"roll_mean_{window}", f"roll_std_{window}"]
    cols += ["day_of_week", "gap_days"]
    return cols
