"""Normal (non-anomalous) transaction generation and shared record helpers.

The helpers here (time, amount, record construction) are reused by the anomaly
injectors so that normal and anomalous rows share an identical schema.
"""

from itertools import count

import numpy as np
import pandas as pd

from app.fraud.features import TYPICAL_AMOUNT
from app.fraud.simulation.config import NORMAL_LABEL, SimulationConfig
from app.fraud.simulation.profiles import PayerProfile, UserProfile

# Diurnal weighting over UTC hours: business hours favoured, small hours rare.
_HOUR_WEIGHTS = np.array(
    [0.2, 0.15, 0.15, 0.15, 0.2, 0.4]      # 00-05 (rare)
    + [1.0, 2.0]                            # 06-07
    + [4.0] * 13                            # 08-20 (busy)
    + [2.0, 1.0, 0.6],                      # 21-23
    dtype=float,
)
_HOUR_PROBS = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

# Monotonic sequence used only as a deterministic tie-breaker when two
# transactions share the exact same timestamp.
_seq_counter = count()


def reset_sequence() -> None:
    """Reset the generation sequence counter (called at the start of each run)."""
    global _seq_counter
    _seq_counter = count()


def _next_seq() -> int:
    return next(_seq_counter)


def sample_time(rng: np.random.Generator, config: SimulationConfig) -> pd.Timestamp:
    """A timestamp inside the window, following the diurnal distribution."""
    n_days = (config.end_ts() - config.start_ts()).days
    day = int(rng.integers(0, n_days))
    hour = int(rng.choice(24, p=_HOUR_PROBS))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return config.start_ts() + pd.Timedelta(days=day, hours=hour, minutes=minute, seconds=second)


def sample_time_in_hours(
    rng: np.random.Generator, config: SimulationConfig, hour_low: int, hour_high: int
) -> pd.Timestamp:
    """A timestamp whose UTC hour falls in ``[hour_low, hour_high)``."""
    n_days = (config.end_ts() - config.start_ts()).days
    day = int(rng.integers(0, n_days))
    hour = int(rng.integers(hour_low, hour_high))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return config.start_ts() + pd.Timedelta(days=day, hours=hour, minutes=minute, seconds=second)


def sample_amount(
    rng: np.random.Generator,
    user: UserProfile,
    currency: str,
    multiplier: float = 1.0,
) -> float:
    """A positive amount in ``currency`` around the user's typical size."""
    base = TYPICAL_AMOUNT[currency] * user.amount_factor
    noise = float(rng.lognormal(mean=0.0, sigma=0.35))
    amount = base * noise * multiplier
    return round(max(amount, 0.01), 2)


def make_record(
    *,
    user_id: int,
    payer: PayerProfile,
    amount: float,
    currency: str,
    payer_name: str,
    origin_country: str,
    occurred_at: pd.Timestamp,
    anomaly_type: str = NORMAL_LABEL,
) -> dict:
    return {
        "user_id": user_id,
        "payer_id": payer.payer_id,
        "amount": float(amount),
        "currency": currency,
        "payer_name": payer_name,
        "origin_country": origin_country,
        "occurred_at": occurred_at,
        "is_anomaly": anomaly_type != NORMAL_LABEL,
        "anomaly_type": anomaly_type,
        "_seq": _next_seq(),
    }


def generate_normal(
    rng: np.random.Generator,
    config: SimulationConfig,
    users: list[UserProfile],
    payers: list[PayerProfile],
    n_normal: int,
) -> list[dict]:
    """Generate exactly ``n_normal`` normal transactions across users."""
    by_id = {p.payer_id: p for p in payers}
    all_ids = np.array([p.payer_id for p in payers])

    weights = np.array([u.activity_weight for u in users], dtype=float)
    probs = weights / weights.sum()
    # Deterministic per-user counts summing to n_normal.
    counts = rng.multinomial(n_normal, probs)

    records: list[dict] = []
    for user, n in zip(users, counts):
        known_ids = np.array(user.known_payer_ids)
        for _ in range(int(n)):
            if rng.random() < config.new_payer_prob:
                payer = by_id[str(rng.choice(all_ids))]
            else:
                payer = by_id[str(rng.choice(known_ids))]
            amount = sample_amount(rng, user, payer.currency)
            records.append(
                make_record(
                    user_id=user.user_id,
                    payer=payer,
                    amount=amount,
                    currency=payer.currency,
                    payer_name=payer.name,
                    origin_country=payer.country,
                    occurred_at=sample_time(rng, config),
                )
            )
    return records
