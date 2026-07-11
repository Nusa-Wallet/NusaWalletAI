"""CLI entry point for the global FX dataset (Phase 8).

Fetches ECB-via-Frankfurter daily rates, computes cross-rates + features + splits, and
writes the raw EUR-based response (data/raw/) and the processed long panel + metadata
(data/processed/). Falls back to a deterministic synthetic provider when offline.

    python -m scripts.fetch_fx_data
    python -m scripts.fetch_fx_data --start 2011-01-01 --end 2026-07-11
    python -m scripts.fetch_fx_data --sample --no-verify
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fx.dataset import FxDatasetConfig  # noqa: E402
from app.fx.dataset.build import build_dataset  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the global FX dataset.")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--sample", action="store_true", help="small offline-friendly config")
    p.add_argument("--no-verify", action="store_true", help="skip cross-rate API cross-check")
    return p.parse_args(argv)


def build_config(args) -> FxDatasetConfig:
    if args.sample:
        return FxDatasetConfig.small_sample()
    overrides = {}
    if args.start:
        overrides["start_date"] = args.start
    if args.end:
        overrides["end_date"] = args.end
    return FxDatasetConfig(**overrides)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    print(f"Building FX dataset {config.start_date}..{config.end()} "
          f"({len(config.pairs())} pairs)...")
    panel, metadata, eur_frame, provenance = build_dataset(config, verify=not args.no_verify)
    print(f"Provider: {provenance['provider']}  rows={len(panel):,}  "
          f"pairs={panel['pair'].nunique()}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = config.dataset_version

    eur_frame.reset_index(names="date").to_parquet(RAW_DIR / f"{stem}_eur_base.parquet", index=False)
    (RAW_DIR / f"{stem}_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    panel.to_parquet(PROCESSED_DIR / f"{stem}.parquet", index=False)
    (PROCESSED_DIR / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nRaw       -> {RAW_DIR / (stem + '_eur_base.parquet')}")
    print(f"Processed -> {PROCESSED_DIR / (stem + '.parquet')}")
    print(f"Metadata  -> {PROCESSED_DIR / (stem + '.metadata.json')}")
    print(f"Date range: {metadata['date_range']['start']} .. {metadata['date_range']['end']}")
    print(f"Splits: {metadata['split_counts']}")
    print(f"Walk-forward windows: {metadata['walk_forward']['count']}")
    ver = metadata["cross_rate_verification"]
    if ver.get("max_rel_error"):
        worst = max(ver["max_rel_error"].values())
        print(f"Cross-rate max rel error (primary): {worst:.2e}")
    print("\nPrimary pairs:")
    for pair, s in metadata["primary_rate_stats"].items():
        print(f"  {pair:<10} obs={s['obs']:<6} last={s['rate_last']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
