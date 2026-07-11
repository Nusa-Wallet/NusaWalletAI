"""Dataset loading, chronological splitting, and feature-matrix extraction."""

from dataclasses import dataclass

import pandas as pd

from app.fraud.feature_spec import BOOL_FEATURES, MODEL_FEATURES

LABEL_COLUMN = "is_anomaly"


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: str) -> pd.DataFrame:
    """Read the synthetic Parquet dataset, sorted chronologically."""
    df = pd.read_parquet(path)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)
    return df.sort_values("occurred_at", kind="mergesort").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> SplitFrames:
    """Split by time into train/val/test with no shuffling and no overlap.

    Boundaries are time quantiles, so the split is purely chronological (train is the
    earliest window, test the latest). This mirrors the proposal's month-based split
    (1-14 / 15-18 / 19-24) on the full 24-month dataset while adapting to any range.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1")
    t = df["occurred_at"]
    train_cut = t.quantile(train_frac)
    val_cut = t.quantile(train_frac + val_frac)
    train = df[t <= train_cut]
    val = df[(t > train_cut) & (t <= val_cut)]
    test = df[t > val_cut]
    return SplitFrames(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the ordered model-feature matrix and integer label vector."""
    x = df[list(MODEL_FEATURES)].copy()
    for col in BOOL_FEATURES:
        x[col] = x[col].astype(int)
    y = df[LABEL_COLUMN].astype(int)
    return x, y
