"""Global NHITS training on the TRAIN split only.

Uses a synthetic contiguous daily index per pair (ECB holiday gaps are irrelevant to a
sequence model) and holds out the tail of TRAIN for early stopping, so the Phase 8
val/test splits are never seen during training or selection.
"""

from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from app.fx.nhits.config import (
    MODEL_NAME,
    NHITS_VERSION,
    QUANTILE_LEVEL,
    SYNTHETIC_FREQ,
    SYNTHETIC_START,
    NhitsTrainConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def to_training_frame(panel: pd.DataFrame, pairs: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Long [unique_id, ds, y] frame from the TRAIN split with a synthetic index."""
    train = panel[panel["split"] == "train"].copy()
    if "source" in train.columns:  # keep a single source per pair if JISDOR was added
        train = train[train["source"] == "ecb"]
    if pairs:
        train = train[train["pair"].isin(pairs)]
    train = train.sort_values(["pair", "date"])

    frames = []
    for pair, group in train.groupby("pair", sort=True):
        ds = pd.date_range(SYNTHETIC_START, periods=len(group), freq=SYNTHETIC_FREQ)
        frames.append(pd.DataFrame({"unique_id": pair, "ds": ds,
                                    "y": group["rate"].to_numpy(dtype=float)}))
    return pd.concat(frames, ignore_index=True)


def train_nhits(config: NhitsTrainConfig, panel: pd.DataFrame):
    """Fit the global NHITS model. Returns (NeuralForecast, training_frame)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS

    frame = to_training_frame(panel, config.pairs)
    model = NHITS(
        h=config.horizon,
        input_size=config.input_size,
        loss=MQLoss(level=[QUANTILE_LEVEL]),
        max_steps=config.max_steps,
        scaler_type=config.scaler_type,
        val_check_steps=config.val_check_steps,
        early_stop_patience_steps=config.early_stop_patience_steps,
        batch_size=config.batch_size,
        windows_batch_size=config.windows_batch_size,
        random_seed=config.seed,
        enable_progress_bar=False,
        accelerator="cpu",
        logger=False,  # no lightning_logs/ dir
    )
    nf = NeuralForecast(models=[model], freq=SYNTHETIC_FREQ)
    nf.fit(frame, val_size=config.val_size)
    return nf, frame


def save_checkpoint(nf, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(directory), overwrite=True, save_dataset=False)


def load_checkpoint(directory: Path):
    """Load a locally-produced (trusted) NHITS checkpoint.

    torch >= 2.6 defaults ``torch.load`` to ``weights_only=True``, which rejects the
    Lightning globals in a NeuralForecast checkpoint. We produced this file ourselves,
    so we load with ``weights_only=False`` for the duration of the call only.
    """
    import torch
    from neuralforecast import NeuralForecast

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        return NeuralForecast.load(path=str(directory))
    finally:
        torch.load = original_load


def training_metadata(config: NhitsTrainConfig, panel: pd.DataFrame, frame: pd.DataFrame,
                      dataset_metadata: dict) -> dict:
    import neuralforecast
    import torch

    return {
        "model_version": NHITS_VERSION,
        "model_name": MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": dataset_metadata.get("dataset_version"),
        "trained_on_split": "train",
        "config": {
            "pairs": "all" if config.pairs is None else list(config.pairs),
            "n_series": int(frame["unique_id"].nunique()),
            "input_size": config.input_size,
            "horizon": config.horizon,
            "max_steps": config.max_steps,
            "val_size": config.val_size,
            "scaler_type": config.scaler_type,
            "quantile_level": QUANTILE_LEVEL,
            "seed": config.seed,
        },
        "train_rows": int(len(frame)),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "neuralforecast": neuralforecast.__version__,
        },
    }
