"""Global NHITS forecasting model for FX (Phase 10).

One NeuralForecast NHITS trained across all currency pairs (pair id = series id), on
the TRAIN split only, with robust scaling, quantile (MQLoss) outputs and early stopping.
The trained checkpoint is served through :class:`NhitsAdapter`, which implements the
Phase 9 ``ForecastModel`` protocol so NHITS is evaluated on the identical walk-forward
windows and metrics as the zero-shot models.
"""

from app.fx.nhits.adapter import NhitsAdapter
from app.fx.nhits.config import NhitsTrainConfig

__all__ = ["NhitsTrainConfig", "NhitsAdapter"]
