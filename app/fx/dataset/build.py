"""Orchestrate the FX dataset build: fetch -> cross-rates -> features -> splits.

Returns the validated long panel plus provenance/metadata (provider, verification,
missing-date report, walk-forward windows). Deterministic given the config and provider.
"""

from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from app.fx.dataset.config import FxDatasetConfig
from app.fx.dataset.crossrates import compute_long_panel
from app.fx.dataset.features import add_features, feature_columns
from app.fx.dataset.providers import fetch_eur_base, verify_cross_rates
from app.fx.dataset.schema import canonical_columns, validate
from app.fx.dataset.splits import assign_split, make_walk_forward_windows

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _missing_dates_report(eur_frame: pd.DataFrame, config: FxDatasetConfig) -> dict:
    """Business days in range with no observation (ECB holidays), documented not filled."""
    expected = pd.bdate_range(config.start_date, config.end(), tz="UTC")
    observed = pd.DatetimeIndex(eur_frame.index)
    missing = expected.difference(observed)
    return {
        "expected_business_days": int(len(expected)),
        "observed_days": int(len(observed)),
        "missing_business_days": int(len(missing)),
        "sample_missing": [d.date().isoformat() for d in missing[:10]],
    }


def build_dataset(config: FxDatasetConfig | None = None, verify: bool = True, fetcher=None):
    """Return (panel, metadata, eur_frame, provenance).

    ``fetcher(config) -> (eur_frame, provenance)`` defaults to the live Frankfurter
    fetch; inject a synthetic one for offline/deterministic tests.
    """
    config = config or FxDatasetConfig()
    eur_frame, provenance = (fetcher or fetch_eur_base)(config)

    panel = compute_long_panel(eur_frame, config.pairs())
    panel = add_features(panel, config)
    panel = assign_split(panel, config)
    panel = panel[canonical_columns(config)]

    validate(panel, config)

    windows = make_walk_forward_windows(panel, config)
    verification = (
        verify_cross_rates(config, eur_frame)
        if verify and provenance["provider"] != "synthetic-fallback"
        else {"method": "skipped", "reason": "synthetic or disabled"}
    )

    split_counts = {k: int(v) for k, v in panel["split"].value_counts().items()}
    metadata = {
        "dataset_version": config.dataset_version,
        "schema_version": config.schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "provenance": provenance,
        "cross_rate_verification": verification,
        "config": {
            "currencies": list(config.currencies),
            "primary_pairs": list(config.primary_pairs),
            "base": config.base,
            "start_date": config.start_date,
            "end_date": config.end(),
            "train_frac": config.train_frac,
            "val_frac": config.val_frac,
        },
        "pairs": {"count": int(panel["pair"].nunique()), "primary": list(config.primary_pairs)},
        "row_count": int(len(panel)),
        "date_range": {
            "start": panel["date"].min().date().isoformat(),
            "end": panel["date"].max().date().isoformat(),
        },
        "obs_per_pair": {
            "min": int(panel.groupby("pair").size().min()),
            "max": int(panel.groupby("pair").size().max()),
        },
        "missing_dates": _missing_dates_report(eur_frame, config),
        "feature_columns": feature_columns(config),
        "split_counts": split_counts,
        "walk_forward": {"count": len(windows), "windows": windows},
        "primary_rate_stats": _primary_stats(panel, config),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return panel, metadata, eur_frame, provenance


def _primary_stats(panel: pd.DataFrame, config: FxDatasetConfig) -> dict:
    stats = {}
    for pair in config.primary_pairs:
        sub = panel[panel["pair"] == pair]
        if len(sub):
            stats[pair] = {
                "obs": int(len(sub)),
                "rate_min": round(float(sub["rate"].min()), 4),
                "rate_max": round(float(sub["rate"].max()), 4),
                "rate_last": round(float(sub.sort_values("date")["rate"].iloc[-1]), 4),
            }
    return stats
