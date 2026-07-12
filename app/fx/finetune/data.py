"""Build (context, target) fine-tuning windows from the TRAIN split only.

Windows are sampled per pair. The latest origins in each pair form the early-stopping
validation set, separated from the training origins by a TARGET_LENGTH gap so a training
target never overlaps a validation context (no leakage). Phase 8 val/test are untouched.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.fx.finetune.config import TARGET_LENGTH, FinetuneConfig


@dataclass
class WindowSet:
    x_train: np.ndarray  # (n_windows, context_length)
    y_train: np.ndarray  # (n_windows, TARGET_LENGTH)
    x_val: np.ndarray
    y_val: np.ndarray

    def sizes(self) -> dict:
        return {"train_windows": int(len(self.x_train)), "val_windows": int(len(self.x_val))}


def build_windows(panel: pd.DataFrame, config: FinetuneConfig) -> WindowSet:
    train = panel[panel["split"] == "train"].copy()
    if "source" in train.columns:
        train = train[train["source"] == "ecb"]
    if config.pairs:
        train = train[train["pair"].isin(config.pairs)]

    pairs = sorted(train["pair"].unique())
    val_span = max(1, math.ceil(config.val_windows / max(len(pairs), 1)))
    L = config.context_length

    train_pool: list[tuple[np.ndarray, np.ndarray]] = []
    val_windows: list[tuple[np.ndarray, np.ndarray]] = []

    for pair in pairs:
        y = train[train["pair"] == pair].sort_values("date")["rate"].to_numpy(dtype=np.float32)
        t_max = len(y) - TARGET_LENGTH  # last valid origin (exclusive upper via +1)
        if t_max <= L:
            continue
        val_lo = max(L, t_max - val_span + 1)
        train_hi = val_lo - TARGET_LENGTH  # gap so train targets don't reach val contexts
        for t in range(L, max(L, train_hi)):
            train_pool.append((y[t - L:t], y[t:t + TARGET_LENGTH]))
        for t in range(val_lo, t_max + 1):
            val_windows.append((y[t - L:t], y[t:t + TARGET_LENGTH]))

    if not train_pool:
        raise ValueError("No training windows — context_length too large for the split")

    rng = np.random.default_rng(config.seed)
    idx = rng.choice(len(train_pool), size=min(config.n_windows, len(train_pool)), replace=False)
    train_sample = [train_pool[i] for i in idx]

    def stack(windows):
        if not windows:
            return np.empty((0, L), np.float32), np.empty((0, TARGET_LENGTH), np.float32)
        xs = np.stack([w[0] for w in windows]).astype(np.float32)
        ys = np.stack([w[1] for w in windows]).astype(np.float32)
        return xs, ys

    x_train, y_train = stack(train_sample)
    x_val, y_val = stack(val_windows)
    return WindowSet(x_train, y_train, x_val, y_val)
