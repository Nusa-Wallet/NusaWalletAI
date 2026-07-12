"""Foundation-model fine-tuning for FX (Phase 11).

Fine-tunes the compact Chronos-Bolt model (chosen for CPU feasibility; the 200M
Chronos-2/TimesFM used in Phase 9 are impractical to fine-tune without a GPU) on the
TRAIN split only, then evaluates zero-shot vs fine-tuned on the same walk-forward windows
as Phase 9/10 via the shared ForecastModel protocol.
"""

from app.fx.finetune.adapter import ChronosBoltAdapter
from app.fx.finetune.config import FinetuneConfig

__all__ = ["FinetuneConfig", "ChronosBoltAdapter"]
