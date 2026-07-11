"""Explainable statistical features for the current FX decision service."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FxStatistics:
    current: float
    ma_7d: float
    volatility_7d: float
    z_score: float


def compute_statistics(series: list[float]) -> FxStatistics:
    values = np.asarray(series, dtype=float)
    current = float(values[-1])
    ma_7d = float(values[-7:].mean())
    volatility = float(values[-7:].std(ddof=0))
    std_all = float(values.std(ddof=0)) or 1e-9
    z_score = (current - float(values.mean())) / std_all
    return FxStatistics(current, ma_7d, volatility, z_score)
