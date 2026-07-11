"""Canonical fraud feature specification — the single source of truth.

Both the training-time batch pass (over the synthetic Parquet dataset) and the
future online inference path build exactly these features, in this order, so a model
trained offline receives an identical vector when served. Grouping follows the
Phase 4 plan: transaction / behavioural / velocity / identity / geographic.
"""

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "transaction": ("amount_idr", "hour_sin", "hour_cos", "day_of_week"),
    "behavioral": ("amount_ratio_user", "amount_zscore_user", "duplicate_similarity"),
    "velocity": ("payer_velocity_10m", "payer_velocity_24h", "user_velocity_24h"),
    "identity": ("is_new_payer", "payer_seen_before", "payer_age_days", "payer_name_quality"),
    "geographic": ("country_seen_before", "currency_seen_before"),
}

# Flat ordered feature list (model input order).
MODEL_FEATURES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)

# Missing-value policy for a user's/payer's first-ever transaction (no history).
# payer_name_quality and the temporal/amount transforms are always computable, so
# they are not listed here.
MISSING_DEFAULTS: dict[str, float | int | bool] = {
    "amount_ratio_user": 1.0,
    "amount_zscore_user": 0.0,
    "duplicate_similarity": 0.0,
    "payer_velocity_10m": 0,
    "payer_velocity_24h": 0,
    "user_velocity_24h": 0,
    "is_new_payer": True,
    "payer_seen_before": False,
    "payer_age_days": 0,
    "country_seen_before": False,
    "currency_seen_before": False,
}

BOOL_FEATURES: frozenset[str] = frozenset(
    {"is_new_payer", "payer_seen_before", "country_seen_before", "currency_seen_before"}
)
INT_FEATURES: frozenset[str] = frozenset(
    {"day_of_week", "payer_velocity_10m", "payer_velocity_24h", "user_velocity_24h", "payer_age_days"}
)


def group_of(feature: str) -> str:
    for group, names in FEATURE_GROUPS.items():
        if feature in names:
            return group
    raise KeyError(feature)
