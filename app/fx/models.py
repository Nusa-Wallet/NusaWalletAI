"""Interfaces for future Chronos, TimesFM, and NHITS forecast adapters."""

from typing import Protocol


class ForecastModel(Protocol):
    @property
    def version(self) -> str: ...

    def forecast(self, series: list[float], horizon_days: int) -> dict: ...
