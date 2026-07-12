"""Phase 9 zero-shot FX backtest CLI, safe to run locally model-by-model.

Examples:
    python scripts/backtest_fx.py --models statistical --max-windows 3 --no-mlflow
    python scripts/backtest_fx.py --models chronos-2 --device cpu --batch-size 2
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.fx.backtest.adapters import create_model  # noqa: E402
from app.fx.backtest.runner import (  # noqa: E402
    BacktestConfig,
    log_mlflow,
    run_backtest,
    save_backtest,
)
from app.fx.dataset.config import DATASET_VERSION, PRIMARY_PAIRS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Phase 9 zero-shot FX backtests.")
    parser.add_argument(
        "--models",
        default="statistical",
        help="comma-separated: statistical, chronos-2, timesfm-2.5",
    )
    parser.add_argument("--pairs", default=",".join(PRIMARY_PAIRS))
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nhits-checkpoint", default=None, help="dir of a trained NHITS checkpoint")
    parser.add_argument("--fee-rate", type=float, default=0.005)
    parser.add_argument("--amount-base", type=float, default=1000.0)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "processed" / f"{DATASET_VERSION}.parquet",
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=ROOT / "data" / "processed" / f"{DATASET_VERSION}.metadata.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "fx_backtests",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_names = [value.strip() for value in args.models.split(",") if value.strip()]
    pairs = tuple(value.strip().upper() for value in args.pairs.split(",") if value.strip())
    horizons = tuple(sorted({int(value) for value in args.horizons.split(",")}))
    config = BacktestConfig(
        pairs=pairs,
        horizons=horizons,
        context_length=args.context_length,
        batch_size=args.batch_size,
        fee_rate=args.fee_rate,
        amount_base=args.amount_base,
        max_windows=args.max_windows,
    )

    print(f"Loading FX panel: {args.dataset}")
    panel = pd.read_parquet(args.dataset)
    metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
    print(f"Models={model_names} pairs={list(pairs)} horizons={list(horizons)}")
    models = [create_model(name, device=args.device, checkpoint=args.nhits_checkpoint)
              for name in model_names]
    predictions, result = run_backtest(panel, metadata, models, config)
    output = save_backtest(predictions, result, args.output_root, args.run_id)

    if not args.no_mlflow:
        if "MLFLOW_TRACKING_URI" not in os.environ:
            db = (args.output_root / "mlflow-phase9.db").resolve().as_posix()
            os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db}"
        log_mlflow(result, output)

    print(f"Predictions: {len(predictions):,}")
    print(f"Cases: {result['case_count']:,}")
    print(f"Output: {output}")
    print("\nSummary:")
    for key, values in result["metrics"].items():
        print(
            f"  {key}: MAE={values['mae']:.4f} "
            f"dir={values['directional_accuracy']:.3f} "
            f"coverage={values['interval_coverage_80']:.3f} "
            f"gain={values['mean_net_gain_vs_immediate']:.2f} "
            f"max_regret={values['maximum_regret']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
