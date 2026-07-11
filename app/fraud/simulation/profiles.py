"""User and payer profile generation.

Profiles are stable given the config seed and are the source of a user's "normal"
behaviour: which payers/countries/currencies they usually see and their typical
transaction size. Anomaly injectors deliberately violate these profiles.
"""

from dataclasses import dataclass

import numpy as np

from app.fraud.simulation.config import (
    COUNTRY_CURRENCY,
    FIRST_NAMES,
    LAST_NAMES,
    NORMAL_COUNTRIES,
    SimulationConfig,
)


@dataclass(frozen=True)
class PayerProfile:
    payer_id: str
    name: str
    country: str
    currency: str


@dataclass(frozen=True)
class UserProfile:
    user_id: int
    amount_factor: float          # scales the currency baseline for this user
    activity_weight: float        # relative share of total transactions
    known_payer_ids: tuple[str, ...]
    usual_countries: frozenset[str]
    usual_currencies: frozenset[str]


def generate_payers(rng: np.random.Generator, config: SimulationConfig) -> list[PayerProfile]:
    payers: list[PayerProfile] = []
    countries = np.array(NORMAL_COUNTRIES)
    for i in range(config.n_payers):
        country = str(rng.choice(countries))
        first = str(rng.choice(FIRST_NAMES))
        last = str(rng.choice(LAST_NAMES))
        payers.append(
            PayerProfile(
                payer_id=f"payer-{i:05d}",
                name=f"{first} {last}",
                country=country,
                currency=COUNTRY_CURRENCY[country],
            )
        )
    return payers


def generate_users(
    rng: np.random.Generator,
    config: SimulationConfig,
    payers: list[PayerProfile],
) -> list[UserProfile]:
    by_id = {p.payer_id: p for p in payers}
    all_ids = np.array([p.payer_id for p in payers])
    users: list[UserProfile] = []
    for i in range(config.n_users):
        # Log-normal spread so users differ by ~1 order of magnitude in size.
        amount_factor = float(rng.lognormal(mean=0.0, sigma=0.5))
        activity_weight = float(rng.lognormal(mean=0.0, sigma=0.6))
        k = int(rng.integers(config.min_known_payers, config.max_known_payers + 1))
        k = min(k, len(all_ids))
        known = rng.choice(all_ids, size=k, replace=False)
        known_ids = tuple(str(pid) for pid in known)
        known_payers = [by_id[pid] for pid in known_ids]
        users.append(
            UserProfile(
                user_id=i + 1,
                amount_factor=amount_factor,
                activity_weight=activity_weight,
                known_payer_ids=known_ids,
                usual_countries=frozenset(p.country for p in known_payers),
                usual_currencies=frozenset(p.currency for p in known_payers),
            )
        )
    return users
