"""Serve a Chronos-Bolt pipeline (zero-shot or fine-tuned) via the ForecastModel protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.fx.backtest.contracts import ForecastBatch
from app.fx.finetune.config import BASE_MODEL_ID, FINETUNE_VERSION, QUANTILE_LEVELS


class ChronosBoltAdapter:
    name = "chronos-bolt"
    version = BASE_MODEL_ID

    def __init__(self, source: str | Path = BASE_MODEL_ID, finetuned: bool = False):
        import torch
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained(
            str(source), device_map="cpu", torch_dtype=torch.float32
        )
        self.name = "chronos-bolt-ft" if finetuned else "chronos-bolt"
        self.version = FINETUNE_VERSION if finetuned else str(source)

    def forecast_batch(self, contexts: list[np.ndarray], horizon: int) -> ForecastBatch:
        import torch

        inputs = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
        quantiles, mean = self.pipe.predict_quantiles(inputs, horizon, list(QUANTILE_LEVELS))
        q = np.asarray(quantiles.numpy())          # (batch, horizon, 3)
        point = np.asarray(mean.numpy())           # (batch, horizon)
        lo, med, hi = q[:, :, 0], q[:, :, 1], q[:, :, 2]
        stacked = np.sort(np.stack([lo, med, hi], axis=0), axis=0)  # enforce monotone
        output = ForecastBatch(
            point=point,
            quantiles={level: stacked[i] for i, level in enumerate(QUANTILE_LEVELS)},
        )
        return output.validate(len(contexts), horizon)
