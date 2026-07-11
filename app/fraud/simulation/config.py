"""Configuration and constants for the fraud transaction simulator.

All randomness is driven by a single seed so that a given ``SimulationConfig``
always produces identical data. The generation window is a fixed calendar range
(not ``today``) so results do not depend on wall-clock time.
"""

from dataclasses import dataclass, replace

import pandas as pd

from app.config import SUPPORTED_CURRENCIES  # noqa: F401 (re-exported for callers)
from app.fraud.features import TYPICAL_AMOUNT

DATASET_VERSION = "fraud-synthetic-v1"
SCHEMA_VERSION = "fraud-txn-schema-1.0.0"

# Static conversion used only to normalise transaction amounts to IDR for
# cross-currency comparability. These are simulation constants, not live rates;
# each currency's TYPICAL_AMOUNT maps to roughly the same IDR baseline (~5M IDR).
RATE_TO_IDR: dict[str, float] = {
    "IDR": 1.0,
    "USD": 16_000.0,
    "SGD": 12_000.0,
    "EUR": 17_500.0,
    "MYR": 3_600.0,
}

# Countries that legitimately transact in the MVP corridors, with their currency.
COUNTRY_CURRENCY: dict[str, str] = {
    "SG": "SGD",
    "US": "USD",
    "DE": "EUR",
    "FR": "EUR",
    "NL": "EUR",
    "MY": "MYR",
}
NORMAL_COUNTRIES: tuple[str, ...] = tuple(COUNTRY_CURRENCY.keys())

# Unusual/high-risk origins used mainly to build country-deviation anomalies.
RISKY_COUNTRY_CURRENCY: dict[str, str] = {
    "NG": "USD",
    "RU": "EUR",
    "UA": "EUR",
    "PK": "USD",
    "KP": "USD",
    "IR": "EUR",
}

# Structuring threshold: transactions are kept just below this IDR value.
STRUCTURING_THRESHOLD_IDR = 50_000_000.0

FIRST_NAMES: tuple[str, ...] = (
    "John", "Mary", "Wei", "Aisyah", "Carlos", "Yuki", "Ahmed", "Sofia",
    "Liam", "Nurul", "Chen", "Priya", "Hans", "Elena", "David", "Fatima",
    "Omar", "Grace", "Kenji", "Lucas",
)
LAST_NAMES: tuple[str, ...] = (
    "Tan", "Smith", "Lee", "Garcia", "Kumar", "Nakamura", "Yusuf", "Muller",
    "Rossi", "Johnson", "Wong", "Silva", "Ali", "Brown", "Lim", "Ivanova",
    "Rahman", "Chen", "Santos", "Kim",
)

ANOMALY_TYPES: tuple[str, ...] = (
    "AMOUNT_SPIKE",
    "VELOCITY_BURST",
    "ODD_HOUR",
    "NEW_PAYER_HIGH_AMOUNT",
    "COUNTRY_DEVIATION",
    "DUPLICATE_PAYMENT",
    "INVALID_IDENTITY",
    "CURRENCY_DEVIATION",
    "ACCOUNT_TAKEOVER",
    "STRUCTURING",
)
NORMAL_LABEL = "NORMAL"


@dataclass(frozen=True)
class SimulationConfig:
    """Fully determines a synthetic dataset given its ``seed``."""

    n_transactions: int = 50_000
    n_users: int = 500
    n_payers: int = 2_000
    months: int = 12
    anomaly_ratio: float = 0.05
    seed: int = 42
    start_date: str = "2025-07-01"  # fixed anchor; not wall-clock dependent

    # Per-user knobs.
    min_known_payers: int = 2
    max_known_payers: int = 8
    # Probability a normal transaction comes from a payer outside the user's
    # known set (a legitimate first-time payer, not an anomaly).
    new_payer_prob: float = 0.08

    dataset_version: str = DATASET_VERSION
    schema_version: str = SCHEMA_VERSION

    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_date, tz="UTC")

    def end_ts(self) -> pd.Timestamp:
        return self.start_ts() + pd.DateOffset(months=self.months)

    @property
    def window_seconds(self) -> float:
        return (self.end_ts() - self.start_ts()).total_seconds()

    @classmethod
    def small_sample(cls, **overrides) -> "SimulationConfig":
        """A fast, still-representative config for tests and smoke runs."""
        base = cls(
            n_transactions=1_500,
            n_users=30,
            n_payers=120,
            months=3,
            anomaly_ratio=0.08,
            seed=42,
        )
        return replace(base, **overrides) if overrides else base
