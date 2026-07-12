"""Light fine-tuning loop for Chronos-Bolt on FX training windows.

Small learning rate, capped steps, and early stopping on a temporally held-out
validation window set — the conservative recipe Phase 11 calls for. The model's own
forward returns the quantile loss, so the loop is a standard optimise/eval cycle.
"""

from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import numpy as np

from app.fx.finetune.config import BASE_MODEL_ID, FINETUNE_VERSION, FinetuneConfig
from app.fx.finetune.data import WindowSet

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def finetune(config: FinetuneConfig, windows: WindowSet):
    """Fine-tune Chronos-Bolt. Returns (pipeline, history dict)."""
    import torch
    from chronos import BaseChronosPipeline

    torch.manual_seed(config.seed)
    pipe = BaseChronosPipeline.from_pretrained(
        config.base_model_id, device_map="cpu", torch_dtype=torch.float32
    )
    model = pipe.model

    frozen = 0
    if config.freeze_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
                frozen += param.numel()
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    x_train = torch.from_numpy(windows.x_train)
    y_train = torch.from_numpy(windows.y_train)
    x_val = torch.from_numpy(windows.x_val)
    y_val = torch.from_numpy(windows.y_val)
    bs = config.batch_size

    def validation_loss() -> float:
        if len(x_val) == 0:
            return float("nan")
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(x_val), bs):
                out = model(context=x_val[i:i + bs], target=y_val[i:i + bs])
                losses.append(float(out.loss))
        model.train()
        return float(np.mean(losses))

    rng = np.random.default_rng(config.seed)
    best_val, best_state, patience, history = float("inf"), None, 0, []
    model.train()
    for step in range(1, config.max_steps + 1):
        idx = rng.integers(0, len(x_train), size=bs)
        out = model(context=x_train[idx], target=y_train[idx])
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            val = validation_loss()
            history.append({"step": step, "train_loss": float(out.loss), "val_loss": val})
            if val == val and val < best_val - 1e-6:  # not NaN and improved
                best_val, patience = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif val == val:
                patience += 1
                if patience >= config.early_stop_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pipe, {"history": history, "best_val_loss": best_val, "steps_run": history[-1]["step"] if history else 0,
                  "frozen_params": int(frozen), "trainable_params": int(sum(p.numel() for p in trainable))}


def save_finetuned(pipe, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pipe.model.save_pretrained(str(directory))


def training_metadata(config: FinetuneConfig, windows: WindowSet, history: dict,
                      dataset_metadata: dict) -> dict:
    import torch
    import chronos

    return {
        "model_version": FINETUNE_VERSION,
        "base_model_id": config.base_model_id or BASE_MODEL_ID,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": dataset_metadata.get("dataset_version"),
        "trained_on_split": "train",
        "config": {
            "pairs": "all" if config.pairs is None else list(config.pairs),
            "context_length": config.context_length,
            "n_windows": config.n_windows,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "freeze_encoder": config.freeze_encoder,
            "seed": config.seed,
        },
        "windows": windows.sizes(),
        "best_val_loss": history["best_val_loss"],
        "steps_run": history["steps_run"],
        "trainable_params": history["trainable_params"],
        "frozen_params": history["frozen_params"],
        "history": history["history"],
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "chronos_forecasting": getattr(chronos, "__version__", "unknown"),
        },
    }
