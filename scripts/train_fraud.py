"""CLI entry point for fraud model training (Phase 5).

Parses arguments and delegates to ``app.fraud.training``; all logic lives under
``app/``. Writes artifacts (CatBoost model, Isolation Forest, ensemble/calibrator,
metadata) and logs experiments to MLflow.

Examples:
    python -m scripts.train_fraud
    python -m scripts.train_fraud --dataset data/synthetic/fraud-synthetic-v1.parquet
    python -m scripts.train_fraud --n-trials 40 --no-mlflow
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fraud.training.pipeline import TrainingConfig, run_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the fraud ensemble model.")
    p.add_argument("--dataset", type=str, default=None, help="path to the synthetic Parquet dataset")
    p.add_argument("--artifacts-dir", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-fpr", type=float, default=0.2)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainingConfig:
    kwargs = dict(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        n_trials=args.n_trials,
        seed=args.seed,
        max_fpr=args.max_fpr,
        mlflow_enabled=not args.no_mlflow,
    )
    if args.dataset:
        kwargs["dataset_path"] = args.dataset
    if args.artifacts_dir:
        kwargs["artifacts_dir"] = args.artifacts_dir
    return TrainingConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    print(f"Training on {config.dataset_path}")
    report = run_training(config)

    print("\nTest-set metrics per experiment:")
    header = f"{'experiment':<18}{'precision':>10}{'recall':>9}{'f1':>8}{'pr_auc':>9}{'roc_auc':>9}{'fpr':>8}{'brier':>8}"
    print(header)
    for name, m in report["experiments"].items():
        print(f"{name:<18}{m['precision']:>10}{m['recall']:>9}{m['f1']:>8}"
              f"{str(m['pr_auc']):>9}{str(m['roc_auc']):>9}{m['fpr']:>8}{m['brier']:>8}")

    print("\nFull-ensemble recall per anomaly type:")
    for t, r in sorted(report["recall_per_anomaly_type"]["full-ensemble"].items()):
        print(f"  {t:<24}{r:>6}")

    dod = report["definition_of_done"]
    print("\nDefinition of done:")
    for k, v in dod.items():
        print(f"  {k:<20}{v}")
    print(f"\nArtifacts -> {report['artifacts_dir']}")
    ok = dod["fpr_below_target"] and dod["beats_rules_f1"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
