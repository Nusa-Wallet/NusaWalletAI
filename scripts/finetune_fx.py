"""Phase 11: light fine-tuning of Chronos-Bolt on FX data (CPU-feasible).

Trains on the TRAIN split only, saves a fine-tuned checkpoint + metadata under
artifacts/fx_finetune/, logs to MLflow. Compare against zero-shot on the same windows:

    python scripts/finetune_fx.py --max-steps 300
    python scripts/backtest_fx.py --models chronos-bolt --no-mlflow          # zero-shot
    python scripts/backtest_fx.py --models chronos-bolt-ft --checkpoint artifacts/fx_finetune/<run>
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.fx.dataset.config import DATASET_VERSION  # noqa: E402
from app.fx.finetune.config import FINETUNE_VERSION, FinetuneConfig  # noqa: E402
from app.fx.finetune.data import build_windows  # noqa: E402
from app.fx.finetune.train import finetune, save_finetuned, training_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune Chronos-Bolt on FX data.")
    p.add_argument("--pairs", default="all")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--n-windows", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / f"{DATASET_VERSION}.parquet")
    p.add_argument("--dataset-metadata", type=Path,
                   default=ROOT / "data" / "processed" / f"{DATASET_VERSION}.metadata.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "fx_finetune")
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pairs = None if args.pairs.strip().lower() == "all" else tuple(
        v.strip().upper() for v in args.pairs.split(",") if v.strip()
    )
    config = FinetuneConfig(
        pairs=pairs, context_length=args.context_length, n_windows=args.n_windows,
        batch_size=args.batch_size, max_steps=args.max_steps, learning_rate=args.lr,
        freeze_encoder=args.freeze_encoder, seed=args.seed,
    )

    panel = pd.read_parquet(args.dataset)
    dataset_metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
    windows = build_windows(panel, config)
    print(f"Windows: {windows.sizes()}  context={config.context_length}")
    print(f"Fine-tuning {config.base_model_id} on CPU (max_steps={config.max_steps})...")

    started = time.time()
    pipe, history = finetune(config, windows)
    elapsed = time.time() - started
    metadata = training_metadata(config, windows, history, dataset_metadata)
    metadata["train_seconds"] = round(elapsed, 1)

    run_id = args.run_id or f"chronos_bolt_ft_seed{config.seed}"
    directory = args.output_root / run_id
    save_finetuned(pipe, directory)
    (directory / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nCheckpoint -> {directory}")
    print(f"Steps run: {history['steps_run']}  best_val_loss: {history['best_val_loss']:.4f}  "
          f"time: {elapsed/60:.1f} min")

    if not args.no_mlflow:
        try:
            import mlflow

            if "MLFLOW_TRACKING_URI" not in os.environ:
                mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').resolve().as_posix()}")
            mlflow.set_experiment("fx-finetune")
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params(metadata["config"])
                mlflow.log_param("model_version", FINETUNE_VERSION)
                mlflow.log_metric("best_val_loss", history["best_val_loss"])
                mlflow.log_artifact(str(directory / "train_metadata.json"))
            print("Logged to MLflow experiment 'fx-finetune'.")
        except Exception as exc:
            print(f"MLflow logging skipped: {exc}")

    print("\nCompare zero-shot vs fine-tuned on the same walk-forward windows:")
    print("  python scripts/backtest_fx.py --models chronos-bolt --no-mlflow")
    print(f"  python scripts/backtest_fx.py --models chronos-bolt-ft --checkpoint {directory} --no-mlflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
