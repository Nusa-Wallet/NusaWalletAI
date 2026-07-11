"""Stable contracts shared by heavyweight adapters and lightweight tests."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

QUANTILE_LEVELS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ForecastBatch:
    """Forecasts for a batch of equally sized horizons."""

    point: np.ndarray
    quantiles: dict[float, np.ndarray]

    def validate(self, batch_size: int, horizon: int) -> "ForecastBatch":
        expected = (batch_size, horizon)
        if np.asarray(self.point).shape != expected:
            raise ValueError(f"Point forecast shape must be {expected}")
        missing = set(QUANTILE_LEVELS) - set(self.quantiles)
        if missing:
            raise ValueError(f"Missing required quantiles: {sorted(missing)}")
        for level, values in self.quantiles.items():
            if np.asarray(values).shape != expected:
                raise ValueError(f"Quantile {level} shape must be {expected}")
        lower = np.asarray(self.quantiles[0.1])
        median = np.asarray(self.quantiles[0.5])
        upper = np.asarray(self.quantiles[0.9])
        if np.any(lower > median) or np.any(median > upper):
            raise ValueError("Forecast quantiles must not cross")
        return self


class ForecastModel(Protocol):
    name: str
    version: str

    def forecast_batch(self, contexts: list[np.ndarray], horizon: int) -> ForecastBatch:
        """Forecast the next horizon observations for each one-dimensional context."""
