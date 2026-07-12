"""Configuration for global NHITS training."""

from dataclasses import dataclass, replace

NHITS_VERSION = "nhits-global-1.0.0"
MODEL_NAME = "NHITS"
QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_LEVEL = 80  # MQLoss level=[80] -> quantiles 0.1 / 0.5 / 0.9

# Synthetic contiguous calendar used internally so ECB holiday gaps never trigger
# frequency filling — NHITS treats each pair as an ordered sequence.
SYNTHETIC_START = "2000-01-01"
SYNTHETIC_FREQ = "D"


@dataclass(frozen=True)
class NhitsTrainConfig:
    pairs: tuple[str, ...] | None = None  # None -> every pair in the dataset (global)
    input_size: int = 180                 # context; Phase 10 sweeps 90/180/365
    horizon: int = 7                      # covers evaluation horizons 1/3/7
    max_steps: int = 300                  # modest for CPU; raise on GPU
    val_size: int = 252                   # tail of TRAIN held out for early stopping
    val_check_steps: int = 50
    early_stop_patience_steps: int = 5
    scaler_type: str = "robust"
    batch_size: int = 32
    windows_batch_size: int = 256
    seed: int = 42

    def with_(self, **overrides) -> "NhitsTrainConfig":
        return replace(self, **overrides)
