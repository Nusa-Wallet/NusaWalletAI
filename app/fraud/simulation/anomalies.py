"""Named anomaly-scenario injectors.

Each injector returns a list of transaction records for a single anomalous *event*
(some scenarios naturally produce several rows, e.g. bursts). Every returned row is
tagged with its ``anomaly_type``. Injectors deliberately violate the user's profile
so the resulting rows are genuinely out-of-pattern relative to their history.
"""

from itertools import count

import numpy as np
import pandas as pd

from app.config import SUPPORTED_CURRENCIES
from app.fraud.simulation.config import (
    COUNTRY_CURRENCY,
    RATE_TO_IDR,
    RISKY_COUNTRY_CURRENCY,
    STRUCTURING_THRESHOLD_IDR,
    SimulationConfig,
)
from app.fraud.simulation.profiles import PayerProfile, UserProfile
from app.fraud.simulation.transactions import (
    make_record,
    sample_amount,
    sample_time,
    sample_time_in_hours,
)

# First country offering each currency, for currency-deviation payers.
_CURRENCY_COUNTRY: dict[str, str] = {}
for _country, _ccy in COUNTRY_CURRENCY.items():
    _CURRENCY_COUNTRY.setdefault(_ccy, _country)

_GIBBERISH = ("", "x", "??", "a", "zzz", "n/a", "-", "test", "12", ".")

_anom_payer_counter = count()


def reset_anomaly_payers() -> None:
    global _anom_payer_counter
    _anom_payer_counter = count()


def _mint_payer(rng: np.random.Generator, country: str, currency: str, name: str) -> PayerProfile:
    return PayerProfile(
        payer_id=f"payer-anom-{next(_anom_payer_counter):06d}",
        name=name,
        country=country,
        currency=currency,
    )


def _pick_user(rng: np.random.Generator, users: list[UserProfile]) -> UserProfile:
    return users[int(rng.integers(0, len(users)))]


def _known_payer(rng: np.random.Generator, user: UserProfile, by_id: dict[str, PayerProfile]) -> PayerProfile:
    pid = str(rng.choice(np.array(user.known_payer_ids)))
    return by_id[pid]


# --- Individual scenarios ---------------------------------------------------

def amount_spike(rng, config, users, by_id):
    user = _pick_user(rng, users)
    payer = _known_payer(rng, user, by_id)
    amount = sample_amount(rng, user, payer.currency, multiplier=float(rng.uniform(6, 15)))
    return [make_record(
        user_id=user.user_id, payer=payer, amount=amount, currency=payer.currency,
        payer_name=payer.name, origin_country=payer.country,
        occurred_at=sample_time(rng, config), anomaly_type="AMOUNT_SPIKE",
    )]


def velocity_burst(rng, config, users, by_id):
    user = _pick_user(rng, users)
    payer = _known_payer(rng, user, by_id)
    n = int(rng.integers(4, 9))
    t0 = sample_time(rng, config)
    offsets = np.sort(rng.integers(0, 600, size=n))  # within a 10-minute window
    rows = []
    for off in offsets:
        rows.append(make_record(
            user_id=user.user_id, payer=payer,
            amount=sample_amount(rng, user, payer.currency),
            currency=payer.currency, payer_name=payer.name, origin_country=payer.country,
            occurred_at=t0 + pd.Timedelta(seconds=int(off)), anomaly_type="VELOCITY_BURST",
        ))
    return rows


def odd_hour(rng, config, users, by_id):
    user = _pick_user(rng, users)
    payer = _known_payer(rng, user, by_id)
    amount = sample_amount(rng, user, payer.currency, multiplier=float(rng.uniform(2, 4)))
    return [make_record(
        user_id=user.user_id, payer=payer, amount=amount, currency=payer.currency,
        payer_name=payer.name, origin_country=payer.country,
        occurred_at=sample_time_in_hours(rng, config, 0, 5), anomaly_type="ODD_HOUR",
    )]


def new_payer_high_amount(rng, config, users, by_id):
    user = _pick_user(rng, users)
    country = str(rng.choice(np.array(list(COUNTRY_CURRENCY.keys()))))
    currency = COUNTRY_CURRENCY[country]
    payer = _mint_payer(rng, country, currency, name="New Client")
    amount = sample_amount(rng, user, currency, multiplier=float(rng.uniform(5, 12)))
    return [make_record(
        user_id=user.user_id, payer=payer, amount=amount, currency=currency,
        payer_name=payer.name, origin_country=country,
        occurred_at=sample_time(rng, config), anomaly_type="NEW_PAYER_HIGH_AMOUNT",
    )]


def country_deviation(rng, config, users, by_id):
    user = _pick_user(rng, users)
    country = str(rng.choice(np.array(list(RISKY_COUNTRY_CURRENCY.keys()))))
    currency = RISKY_COUNTRY_CURRENCY[country]
    payer = _mint_payer(rng, country, currency, name="Remote Payer")
    amount = sample_amount(rng, user, currency)
    return [make_record(
        user_id=user.user_id, payer=payer, amount=amount, currency=currency,
        payer_name=payer.name, origin_country=country,
        occurred_at=sample_time(rng, config), anomaly_type="COUNTRY_DEVIATION",
    )]


def duplicate_payment(rng, config, users, by_id):
    user = _pick_user(rng, users)
    payer = _known_payer(rng, user, by_id)
    amount = sample_amount(rng, user, payer.currency)
    t0 = sample_time(rng, config)
    n = int(rng.integers(2, 4))
    rows = []
    for j in range(n):
        rows.append(make_record(
            user_id=user.user_id, payer=payer, amount=amount, currency=payer.currency,
            payer_name=payer.name, origin_country=payer.country,
            occurred_at=t0 + pd.Timedelta(seconds=int(rng.integers(1, 180))),
            anomaly_type="DUPLICATE_PAYMENT",
        ))
    return rows


def invalid_identity(rng, config, users, by_id):
    user = _pick_user(rng, users)
    country = str(rng.choice(np.array(list(COUNTRY_CURRENCY.keys()))))
    currency = COUNTRY_CURRENCY[country]
    name = str(rng.choice(np.array(_GIBBERISH, dtype=object)))
    payer = _mint_payer(rng, country, currency, name=name)
    amount = sample_amount(rng, user, currency, multiplier=float(rng.uniform(1, 4)))
    return [make_record(
        user_id=user.user_id, payer=payer, amount=amount, currency=currency,
        payer_name=name, origin_country=country,
        occurred_at=sample_time(rng, config), anomaly_type="INVALID_IDENTITY",
    )]


def currency_deviation(rng, config, users, by_id):
    user = _pick_user(rng, users)
    options = [c for c in SUPPORTED_CURRENCIES if c != "IDR" and c not in user.usual_currencies]
    if not options:
        options = [c for c in SUPPORTED_CURRENCIES if c != "IDR"]
    currency = str(rng.choice(np.array(sorted(options))))
    country = _CURRENCY_COUNTRY.get(currency, "US")
    payer = _mint_payer(rng, country, currency, name="Overseas Client")
    amount = sample_amount(rng, user, currency, multiplier=float(rng.uniform(1, 3)))
    return [make_record(
        user_id=user.user_id, payer=payer, amount=amount, currency=currency,
        payer_name=payer.name, origin_country=country,
        occurred_at=sample_time(rng, config), anomaly_type="CURRENCY_DEVIATION",
    )]


def account_takeover(rng, config, users, by_id):
    user = _pick_user(rng, users)
    n = int(rng.integers(3, 6))
    base = sample_time_in_hours(rng, config, 0, 5)
    risky = list(RISKY_COUNTRY_CURRENCY.items())
    rows = []
    for j in range(n):
        country, currency = risky[int(rng.integers(0, len(risky)))]
        payer = _mint_payer(rng, country, currency, name="Unknown")
        amount = sample_amount(rng, user, currency, multiplier=float(rng.uniform(3, 8)) * (j + 1))
        rows.append(make_record(
            user_id=user.user_id, payer=payer, amount=amount, currency=currency,
            payer_name=payer.name, origin_country=country,
            occurred_at=base + pd.Timedelta(minutes=int(rng.integers(0, 120))),
            anomaly_type="ACCOUNT_TAKEOVER",
        ))
    return rows


def structuring(rng, config, users, by_id):
    user = _pick_user(rng, users)
    payer = _known_payer(rng, user, by_id)
    currency = payer.currency
    n = int(rng.integers(4, 8))
    t0 = sample_time(rng, config)
    rows = []
    for j in range(n):
        target_idr = STRUCTURING_THRESHOLD_IDR * float(rng.uniform(0.85, 0.98))
        amount = round(target_idr / RATE_TO_IDR[currency], 2)
        rows.append(make_record(
            user_id=user.user_id, payer=payer, amount=amount, currency=currency,
            payer_name=payer.name, origin_country=payer.country,
            occurred_at=t0 + pd.Timedelta(hours=int(rng.integers(0, 48))),
            anomaly_type="STRUCTURING",
        ))
    return rows


# Ordered so round-robin dispatch guarantees every scenario is represented.
INJECTORS = (
    amount_spike,
    velocity_burst,
    odd_hour,
    new_payer_high_amount,
    country_deviation,
    duplicate_payment,
    invalid_identity,
    currency_deviation,
    account_takeover,
    structuring,
)


def inject_anomalies(
    rng: np.random.Generator,
    config: SimulationConfig,
    users: list[UserProfile],
    by_id: dict[str, PayerProfile],
    target_rows: int,
) -> list[dict]:
    """Round-robin over scenarios until at least ``target_rows`` anomalies exist.

    Round-robin ordering guarantees every scenario appears at least once (given a
    non-trivial target), satisfying the "every anomaly type represented" requirement.
    """
    reset_anomaly_payers()
    records: list[dict] = []
    idx = 0
    n_scenarios = len(INJECTORS)
    # Ensure at least one full sweep so no scenario is missing.
    while len(records) < target_rows or idx < n_scenarios:
        injector = INJECTORS[idx % n_scenarios]
        records.extend(injector(rng, config, users, by_id))
        idx += 1
    return records
