"""Top-level orchestration: profiles -> normal + anomalies -> features -> dataset.

``generate_dataset`` is fully deterministic for a given :class:`SimulationConfig`:
one seeded RNG drives every draw, records are sorted with a stable tie-breaker, and
transaction ids are assigned only after sorting.
"""

from datetime import datetime, timezone
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from app.fraud.simulation.anomalies import inject_anomalies
from app.fraud.simulation.config import RATE_TO_IDR, SimulationConfig
from app.fraud.simulation.features import compute_historical_features
from app.fraud.simulation.profiles import generate_payers, generate_users
from app.fraud.simulation.schema import CANONICAL_COLUMNS, validate
from app.fraud.simulation.transactions import generate_normal, reset_sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def generate_dataset(config: SimulationConfig) -> tuple[pd.DataFrame, dict]:
    """Return the labelled dataset and its provenance metadata."""
    reset_sequence()
    rng = np.random.default_rng(config.seed)

    payers = generate_payers(rng, config)
    users = generate_users(rng, config, payers)
    by_id = {p.payer_id: p for p in payers}

    target_anomaly_rows = round(config.n_transactions * config.anomaly_ratio)
    anomalies = inject_anomalies(rng, config, users, by_id, target_anomaly_rows)
    n_anomaly = len(anomalies)
    n_normal = max(config.n_transactions - n_anomaly, 0)
    normals = generate_normal(rng, config, users, payers, n_normal)

    df = pd.DataFrame.from_records(normals + anomalies)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)

    # Deterministic chronological order; ``_seq`` breaks exact timestamp ties.
    df = df.sort_values(["occurred_at", "_seq"], kind="mergesort").reset_index(drop=True)
    df["transaction_id"] = [f"trx-{i:08d}" for i in range(len(df))]
    df["amount_idr"] = (df["amount"] * df["currency"].map(RATE_TO_IDR)).round(2)

    features = compute_historical_features(df)
    df = pd.concat([df.drop(columns="_seq"), features], axis=1)
    df = df[list(CANONICAL_COLUMNS)]

    metadata = _build_metadata(df, config)
    return df, metadata


def _build_metadata(df: pd.DataFrame, config: SimulationConfig) -> dict:
    anomaly_counts = df["anomaly_type"].value_counts().sort_index().to_dict()
    n_anomaly = int(df["is_anomaly"].sum())
    return {
        "dataset_version": config.dataset_version,
        "schema_version": config.schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": config.seed,
        "config": {
            "n_transactions": config.n_transactions,
            "n_users": config.n_users,
            "n_payers": config.n_payers,
            "months": config.months,
            "anomaly_ratio": config.anomaly_ratio,
            "start_date": config.start_date,
        },
        "row_count": int(len(df)),
        "date_range": {
            "start": df["occurred_at"].min().isoformat(),
            "end": df["occurred_at"].max().isoformat(),
        },
        "anomaly": {
            "count": n_anomaly,
            "ratio_target": config.anomaly_ratio,
            "ratio_actual": round(n_anomaly / len(df), 4),
            "type_counts": {str(k): int(v) for k, v in anomaly_counts.items()},
        },
        "currency_counts": {str(k): int(v) for k, v in df["currency"].value_counts().items()},
        "amount_idr_stats": {
            "min": float(df["amount_idr"].min()),
            "max": float(df["amount_idr"].max()),
            "mean": round(float(df["amount_idr"].mean()), 2),
            "median": round(float(df["amount_idr"].median()), 2),
        },
        "unique_transaction_ids": int(df["transaction_id"].nunique()),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def generate_and_validate(config: SimulationConfig) -> tuple[pd.DataFrame, dict]:
    """Generate and run Pandera validation before returning."""
    df, metadata = generate_dataset(config)
    validate(df, config)
    return df, metadata
