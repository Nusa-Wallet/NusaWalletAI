"""Train the global NHITS FX model on CPU (Phase 10).

Trains on the TRAIN split only (val/test stay untouched), saves a checkpoint + metadata
under artifacts/fx_nhits/, and logs to MLflow. Evaluate it afterwards with the Phase 9
backtest for a fair comparison:

    python scripts/train_nhits.py --input-size 180 --max-steps 300
    python scripts/backtest_fx.py --models nhits --nhits-checkpoint artifacts/fx_nhits/<run>
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.fx.dataset.config import DATASET_VERSION, PRIMARY_PAIRS  # noqa: E402
from app.fx.nhits.config import NHITS_VERSION, NhitsTrainConfig  # noqa: E402
from app.fx.nhits.training import save_checkpoint, train_nhits, training_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the global NHITS FX model.")
    p.add_argument("--pairs", default="all", help="'all' or comma-separated pairs")
    p.add_argument("--input-size", type=int, default=180)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--val-size", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / f"{DATASET_VERSION}.parquet")
    p.add_argument("--dataset-metadata", type=Path,
                   default=ROOT / "data" / "processed" / f"{DATASET_VERSION}.metadata.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "fx_nhits")
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pairs = None if args.pairs.strip().lower() == "all" else tuple(
        v.strip().upper() for v in args.pairs.split(",") if v.strip()
    )
    config = NhitsTrainConfig(
        pairs=pairs, input_size=args.input_size, horizon=args.horizon,
        max_steps=args.max_steps, val_size=args.val_size, seed=args.seed,
    )

    panel = pd.read_parquet(args.dataset)
    dataset_metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
    print(f"Training global NHITS: input_size={config.input_size} horizon={config.horizon} "
          f"max_steps={config.max_steps} pairs={'all' if pairs is None else len(pairs)}")

    nf, frame = train_nhits(config, panel)
    metadata = training_metadata(config, panel, frame, dataset_metadata)

    run_id = args.run_id or f"nhits_ctx{config.input_size}_seed{config.seed}"
    directory = args.output_root / run_id
    save_checkpoint(nf, directory)
    (directory / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Checkpoint -> {directory}")
    print(f"Series trained: {metadata['config']['n_series']}  rows: {metadata['train_rows']:,}")

    if not args.no_mlflow:
        try:
            import mlflow

            if "MLFLOW_TRACKING_URI" not in os.environ:
                db = (ROOT / "mlflow.db").resolve().as_posix()
                mlflow.set_tracking_uri(f"sqlite:///{db}")
            mlflow.set_experiment("fx-nhits")
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params(metadata["config"])
                mlflow.log_param("model_version", NHITS_VERSION)
                mlflow.log_artifact(str(directory / "train_metadata.json"))
            print("Logged to MLflow experiment 'fx-nhits'.")
        except Exception as exc:
            print(f"MLflow logging skipped: {exc}")

    print("\nNext: evaluate on the same walk-forward windows as Phase 9:")
    print(f"  python scripts/backtest_fx.py --models nhits --nhits-checkpoint {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
