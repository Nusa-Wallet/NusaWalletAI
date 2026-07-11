"""CLI entry point for the synthetic fraud dataset (Phase 3).

Parses arguments and delegates to ``app.fraud.simulation``; all generation logic
lives under ``app/``. Outputs a Parquet dataset plus a metadata JSON recording seed,
provenance, and distributions.

Examples:
    python -m scripts.generate_fraud_data                 # full 50k dataset
    python -m scripts.generate_fraud_data --sample        # fast small dataset
    python -m scripts.generate_fraud_data --rows 200000 --users 3000 --months 24
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/generate_fraud_data.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fraud.simulation import SimulationConfig, generate_dataset  # noqa: E402
from app.fraud.simulation.schema import validate  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic fraud dataset.")
    parser.add_argument("--rows", type=int, default=50_000, help="total transactions")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--payers", type=int, default=2_000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2025-07-01")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample", action="store_true", help="use the small_sample preset")
    parser.add_argument("--no-validate", action="store_true", help="skip Pandera validation")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimulationConfig:
    if args.sample:
        return SimulationConfig.small_sample(seed=args.seed)
    return SimulationConfig(
        n_transactions=args.rows,
        n_users=args.users,
        n_payers=args.payers,
        months=args.months,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
        start_date=args.start_date,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    print(f"Generating {config.n_transactions:,} transactions (seed={config.seed})...")
    df, metadata = generate_dataset(config)

    if not args.no_validate:
        print("Validating against canonical schema...")
        validate(df, config)
        print("Validation passed.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = config.dataset_version
    parquet_path = args.out_dir / f"{stem}.parquet"
    meta_path = args.out_dir / f"{stem}.metadata.json"

    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {len(df):,} rows -> {parquet_path}")
    print(f"Metadata            -> {meta_path}")
    anomaly = metadata["anomaly"]
    print(f"Anomalies: {anomaly['count']:,} ({anomaly['ratio_actual']:.2%})")
    for name, count in sorted(anomaly["type_counts"].items()):
        print(f"  {name:<24} {count:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
