"""Global FX dataset pipeline (Phase 8).

Builds a reproducible long-panel dataset of daily cross-currency rates from ECB
reference rates (retrieved via Frankfurter), with log-returns, lag/rolling features,
chronological splits, and walk-forward windows for later forecasting/backtesting.

Real market data with a deterministic synthetic fallback so the pipeline and tests run
offline. Provider provenance is always recorded so real and fallback data are never
confused.
"""

from app.fx.dataset.build import build_dataset
from app.fx.dataset.config import FxDatasetConfig

__all__ = ["FxDatasetConfig", "build_dataset"]
