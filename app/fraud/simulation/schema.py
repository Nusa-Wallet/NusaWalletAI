"""Canonical dataset schema and Pandera validation.

``CANONICAL_COLUMNS`` fixes the column order of every generated dataset. The Pandera
schema enforces the "definition of done" invariants: unique IDs, positive amounts,
supported currencies, valid timestamps/hours, and known anomaly labels.
"""

import pandas as pd

try:  # pandera >= 0.20 splits the pandas API into a submodule
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - older pandera fallback
    import pandera as pa

from app.config import SUPPORTED_CURRENCIES
from app.fraud.simulation.config import ANOMALY_TYPES, NORMAL_LABEL, SimulationConfig

CANONICAL_COLUMNS: tuple[str, ...] = (
    # identity & raw transaction
    "transaction_id",
    "user_id",
    "payer_id",
    "occurred_at",
    "amount",
    "currency",
    "amount_idr",
    "payer_name",
    "origin_country",
    # past-only historical / behavioural features
    "hour",
    "day_of_week",
    "is_new_payer",
    "payer_seen_before",
    "payer_age_days",
    "country_seen_before",
    "currency_seen_before",
    "payer_velocity_10m",
    "payer_velocity_24h",
    "user_velocity_24h",
    "amount_ratio_user",
    "amount_zscore_user",
    # ground-truth labels
    "is_anomaly",
    "anomaly_type",
)

_ALLOWED_LABELS = list(ANOMALY_TYPES) + [NORMAL_LABEL]
_ALLOWED_COUNTRY = pa.Check.str_length(2, 2)


def build_schema(config: SimulationConfig) -> "pa.DataFrameSchema":
    start, end = config.start_ts(), config.end_ts()
    return pa.DataFrameSchema(
        {
            "transaction_id": pa.Column(str, unique=True, nullable=False),
            "user_id": pa.Column(int, pa.Check.in_range(1, config.n_users)),
            "payer_id": pa.Column(str, nullable=False),
            "occurred_at": pa.Column(
                "datetime64[ns, UTC]",
                pa.Check(lambda s: (s >= start) & (s <= end), error="timestamp out of window"),
            ),
            "amount": pa.Column(float, pa.Check.gt(0)),
            "currency": pa.Column(str, pa.Check.isin(sorted(SUPPORTED_CURRENCIES))),
            "amount_idr": pa.Column(float, pa.Check.gt(0)),
            "payer_name": pa.Column(str, nullable=False),
            "origin_country": pa.Column(str, _ALLOWED_COUNTRY),
            "hour": pa.Column(int, pa.Check.in_range(0, 23)),
            "day_of_week": pa.Column(int, pa.Check.in_range(0, 6)),
            "is_new_payer": pa.Column(bool),
            "payer_seen_before": pa.Column(bool),
            "payer_age_days": pa.Column(int, pa.Check.ge(0)),
            "country_seen_before": pa.Column(bool),
            "currency_seen_before": pa.Column(bool),
            "payer_velocity_10m": pa.Column(int, pa.Check.ge(0)),
            "payer_velocity_24h": pa.Column(int, pa.Check.ge(0)),
            "user_velocity_24h": pa.Column(int, pa.Check.ge(0)),
            "amount_ratio_user": pa.Column(float, pa.Check.gt(0)),
            "amount_zscore_user": pa.Column(float),
            "is_anomaly": pa.Column(bool),
            "anomaly_type": pa.Column(str, pa.Check.isin(_ALLOWED_LABELS)),
        },
        strict=True,
        ordered=True,
        coerce=True,
    )


def validate(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """Validate ``df`` against the canonical schema, collecting all failures."""
    return build_schema(config).validate(df, lazy=True)
