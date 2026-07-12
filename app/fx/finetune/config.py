"""Configuration for Chronos-Bolt fine-tuning."""

from dataclasses import dataclass, replace

FINETUNE_VERSION = "chronos-bolt-ft-1.0.0"
BASE_MODEL_ID = "amazon/chronos-bolt-small"
TARGET_LENGTH = 64  # Chronos-Bolt's native prediction head length
QUANTILE_LEVELS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class FinetuneConfig:
    pairs: tuple[str, ...] | None = None  # None -> all pairs (global fine-tune)
    base_model_id: str = BASE_MODEL_ID
    context_length: int = 512
    n_windows: int = 4000        # sampled (context, target) training windows
    val_windows: int = 500       # held out from the TRAIN tail for early stopping
    batch_size: int = 8
    max_steps: int = 300
    learning_rate: float = 1e-5  # small LR — light fine-tuning
    freeze_encoder: bool = False # optional parameter-efficient variant
    early_stop_patience: int = 4
    eval_every: int = 25
    seed: int = 42

    def with_(self, **overrides) -> "FinetuneConfig":
        return replace(self, **overrides)
