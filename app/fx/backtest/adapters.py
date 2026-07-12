"""Lazy adapters for Phase 9 models.

Heavy libraries are imported only when their adapter is instantiated, keeping local
tests and the FastAPI service independent from PyTorch/foundation-model packages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.fx.backtest.contracts import ForecastBatch, QUANTILE_LEVELS


def _empirical_intervals(context: np.ndarray, point: np.ndarray) -> dict[float, np.ndarray]:
    """Comparator intervals from historical log-return dispersion."""
    clean = np.asarray(context, dtype=float)
    returns = np.diff(np.log(clean[-min(len(clean), 252):]))
    sigma = float(np.nanstd(returns)) if len(returns) else 0.0
    steps = np.arange(1, len(point) + 1, dtype=float)
    spread = np.asarray(point) * sigma * np.sqrt(steps) * 1.2815515655446004
    return {0.1: point - spread, 0.5: point.copy(), 0.9: point + spread}


class StatisticalComparator:
    """Drift forecast used only as the required scientific comparator."""

    name = "statistical-drift"
    version = "1.0.0"

    def forecast_batch(self, contexts: list[np.ndarray], horizon: int) -> ForecastBatch:
        points, quantiles = [], {level: [] for level in QUANTILE_LEVELS}
        for values in contexts:
            context = np.asarray(values, dtype=float)
            if len(context) < 2 or np.any(context <= 0):
                raise ValueError("Comparator needs at least two positive observations")
            lookback = context[-min(21, len(context)):]
            daily_log_drift = float(np.mean(np.diff(np.log(lookback))))
            steps = np.arange(1, horizon + 1, dtype=float)
            point = context[-1] * np.exp(daily_log_drift * steps)
            intervals = _empirical_intervals(context, point)
            points.append(point)
            for level in QUANTILE_LEVELS:
                quantiles[level].append(intervals[level])
        output = ForecastBatch(
            point=np.asarray(points),
            quantiles={level: np.asarray(rows) for level, rows in quantiles.items()},
        )
        return output.validate(len(contexts), horizon)


class Chronos2Adapter:
    name = "chronos-2"
    version = "amazon/chronos-2"

    def __init__(self, device: str = "cpu", model_id: str | None = None):
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Chronos is unavailable; install requirements-neural.txt locally"
            ) from exc
        self.model_id = model_id or self.version
        self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=device)

    def forecast_batch(self, contexts: list[np.ndarray], horizon: int) -> ForecastBatch:
        frames = []
        for index, values in enumerate(contexts):
            values = np.asarray(values, dtype=float)
            timestamps = pd.date_range("2000-01-01", periods=len(values), freq="D")
            frames.append(pd.DataFrame({"id": index, "timestamp": timestamps, "target": values}))
        prediction = self.pipeline.predict_df(
            pd.concat(frames, ignore_index=True),
            prediction_length=horizon,
            quantile_levels=list(QUANTILE_LEVELS),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        def matrix(column: str) -> np.ndarray:
            ordered = prediction.sort_values(["id", "timestamp"])
            rows = [
                group[column].to_numpy(dtype=float)
                for _, group in ordered.groupby("id", sort=False)
            ]
            return np.asarray(rows)

        output = ForecastBatch(
            point=matrix("predictions"),
            quantiles={level: matrix(str(level)) for level in QUANTILE_LEVELS},
        )
        return output.validate(len(contexts), horizon)


class TimesFmAdapter:
    name = "timesfm-2.5"
    version = "google/timesfm-2.5-200m-pytorch"

    def __init__(
        self,
        device: str = "cpu",
        model_id: str | None = None,
        max_context: int = 2048,
        max_horizon: int = 32,
    ):
        try:
            import timesfm
        except ImportError as exc:
            raise RuntimeError(
                "TimesFM is unavailable; install requirements-neural.txt locally"
            ) from exc
        self.model_id = model_id or self.version
        self._timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self.device = device

    def forecast_batch(self, contexts: list[np.ndarray], horizon: int) -> ForecastBatch:
        point, raw_quantiles = self.model.forecast(
            horizon=horizon,
            inputs=[np.asarray(values, dtype=np.float32) for values in contexts],
        )
        point = np.asarray(point, dtype=float)
        raw = np.asarray(raw_quantiles, dtype=float)
        # Official TimesFM 2.5 order: mean, then q0.1 through q0.9.
        positions = {0.1: 1, 0.5: 5, 0.9: 9}
        output = ForecastBatch(
            point=point,
            quantiles={level: raw[:, :, position] for level, position in positions.items()},
        )
        return output.validate(len(contexts), horizon)


def create_model(name: str, device: str = "cpu", checkpoint: str | None = None):
    normalized = name.strip().lower()
    if normalized in {"statistical", "statistical-drift", "baseline"}:
        return StatisticalComparator()
    if normalized in {"chronos", "chronos-2"}:
        return Chronos2Adapter(device=device)
    if normalized in {"timesfm", "timesfm-2.5"}:
        return TimesFmAdapter(device=device)
    if normalized in {"nhits", "nhits-global"}:
        if not checkpoint:
            raise ValueError("nhits requires a checkpoint (pass --nhits-checkpoint)")
        from app.fx.nhits import NhitsAdapter

        return NhitsAdapter(checkpoint)
    if normalized in {"chronos-bolt", "chronos-bolt-zero"}:
        from app.fx.finetune import ChronosBoltAdapter

        return ChronosBoltAdapter(finetuned=False)  # zero-shot baseline
    if normalized in {"chronos-bolt-ft", "chronos-bolt-finetuned"}:
        if not checkpoint:
            raise ValueError("chronos-bolt-ft requires a checkpoint (pass --checkpoint)")
        from app.fx.finetune import ChronosBoltAdapter

        return ChronosBoltAdapter(checkpoint, finetuned=True)
    raise ValueError(f"Unknown FX model: {name}")
