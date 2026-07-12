"""Serve a trained global NHITS checkpoint through the Phase 9 ForecastModel protocol.

Each rolling-origin context is turned into a one-series frame with a synthetic
contiguous index; NeuralForecast uses the last ``input_size`` points to forecast the
next ``horizon`` steps. Quantiles are sorted per step to guarantee no crossing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.fx.backtest.contracts import ForecastBatch, QUANTILE_LEVELS
from app.fx.nhits.config import NHITS_VERSION, SYNTHETIC_FREQ, SYNTHETIC_START


class NhitsAdapter:
    name = "nhits-global"
    version = NHITS_VERSION

    def __init__(self, checkpoint_dir: str | Path):
        from app.fx.nhits.training import load_checkpoint

        self.checkpoint_dir = str(checkpoint_dir)
        self.nf = load_checkpoint(Path(checkpoint_dir))

    @staticmethod
    def _detect_columns(columns) -> dict[str, str]:
        cols = list(columns)
        point = next((c for c in cols if c.endswith("-median")), None)
        lower = next((c for c in cols if "-lo-" in c), None)
        upper = next((c for c in cols if "-hi-" in c), None)
        if not (point and lower and upper):
            raise RuntimeError(f"Unexpected NHITS predict columns: {cols}")
        return {"point": point, "lo": lower, "hi": upper}

    def forecast_batch(self, contexts: list[np.ndarray], horizon: int) -> ForecastBatch:
        frames = []
        for index, values in enumerate(contexts):
            series = np.asarray(values, dtype=float)
            ds = pd.date_range(SYNTHETIC_START, periods=len(series), freq=SYNTHETIC_FREQ)
            frames.append(pd.DataFrame({"unique_id": str(index), "ds": ds, "y": series}))

        prediction = self.nf.predict(df=pd.concat(frames, ignore_index=True))
        if "unique_id" not in prediction.columns:  # current NF returns it as the index
            prediction = prediction.reset_index()
        prediction["_uid"] = prediction["unique_id"].astype(int)
        prediction = prediction.sort_values(["_uid", "ds"])
        cols = self._detect_columns(prediction.columns)

        def matrix(column: str) -> np.ndarray:
            rows = [g[column].to_numpy(dtype=float)[:horizon]
                    for _, g in prediction.groupby("_uid", sort=True)]
            return np.asarray(rows)

        lo, median, hi = matrix(cols["lo"]), matrix(cols["point"]), matrix(cols["hi"])
        # Enforce monotone quantiles per element (isotonic fix -> no crossing).
        stacked = np.sort(np.stack([lo, median, hi], axis=0), axis=0)
        quantiles = {level: stacked[i] for i, level in enumerate(QUANTILE_LEVELS)}
        output = ForecastBatch(point=stacked[1], quantiles=quantiles)
        return output.validate(len(contexts), horizon)
