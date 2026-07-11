"""Canonical transparent fraud rules — the single source shared by the training
ensemble and the explainability layer.

Each rule has a vectorised mask (for scoring whole frames) and a scalar Indonesian
template (for per-transaction explanation), so a triggered rule always has a
human-readable reason attached. Rule scores follow the conservative "strongest
triggered signal" policy.

Phase 6 note: the ``high_risk_country`` rule was added to address the Phase 5 flag
(weak country/currency-deviation recall). It uses a general high-risk-jurisdiction
list (FATF-style), not the data generator's internal set, so it is a legitimate AML
signal rather than a way to "learn the simulator".
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# General high-risk / sanctioned jurisdictions (illustrative FATF-style list).
HIGH_RISK_COUNTRIES: frozenset[str] = frozenset(
    {"KP", "IR", "RU", "SY", "NG", "PK", "UA", "AF", "MM", "YE"}
)


@dataclass(frozen=True)
class RuleDef:
    code: str
    weight: float
    topic: str
    mask: Callable[[pd.DataFrame], np.ndarray]
    template: Callable[[Mapping], str]


@dataclass(frozen=True)
class RuleHit:
    code: str
    weight: float
    topic: str
    message: str


RULE_DEFS: tuple[RuleDef, ...] = (
    RuleDef(
        "high_amount_ratio", 0.85, "amount",
        lambda d: d["amount_ratio_user"].to_numpy() > 5,
        lambda r: f"Nominal transaksi sekitar {float(r['amount_ratio_user']):.1f}x lebih besar dari pola normal pengguna.",
    ),
    RuleDef(
        "high_amount_zscore", 0.75, "amount",
        lambda d: d["amount_zscore_user"].to_numpy() > 4,
        lambda r: "Nominal transaksi menyimpang jauh dari kebiasaan pengguna.",
    ),
    RuleDef(
        "odd_hour", 0.60, "odd_hour",
        lambda d: d["hour"].to_numpy() < 6,
        lambda r: f"Transaksi terjadi pada jam tidak biasa (pukul {int(r['hour']):02d}:00 dini hari).",
    ),
    RuleDef(
        "low_name_quality", 0.75, "identity_name",
        lambda d: d["payer_name_quality"].to_numpy() < 0.3,
        lambda r: "Nama pembayar tidak lengkap atau tidak wajar.",
    ),
    RuleDef(
        "payer_velocity_burst", 0.80, "velocity",
        lambda d: d["payer_velocity_10m"].to_numpy() >= 3,
        lambda r: f"Terdapat {int(r['payer_velocity_10m'])} transaksi dari pembayar ini dalam 10 menit terakhir.",
    ),
    RuleDef(
        "duplicate_payment", 0.70, "duplicate",
        lambda d: d["duplicate_similarity"].to_numpy() > 0.9,
        lambda r: "Transaksi sangat menyerupai pembayaran sebelumnya (indikasi duplikasi).",
    ),
    RuleDef(
        "new_payer_high_amount", 0.70, "new_payer",
        lambda d: (~d["payer_seen_before"].to_numpy().astype(bool)) & (d["amount_ratio_user"].to_numpy() > 3),
        lambda r: "Pembayar baru langsung melakukan transaksi bernominal besar.",
    ),
    RuleDef(
        "high_risk_country", 0.70, "country",
        lambda d: d["origin_country"].isin(HIGH_RISK_COUNTRIES).to_numpy(),
        lambda r: f"Negara asal ({r['origin_country']}) termasuk yurisdiksi berisiko tinggi.",
    ),
    RuleDef(
        "currency_deviation", 0.45, "currency",
        lambda d: ~d["currency_seen_before"].to_numpy().astype(bool),
        lambda r: f"Mata uang ({r['currency']}) tidak biasa digunakan pengguna.",
    ),
    RuleDef(
        "country_deviation", 0.45, "country",
        lambda d: ~d["country_seen_before"].to_numpy().astype(bool),
        lambda r: f"Negara asal ({r['origin_country']}) belum pernah digunakan pengguna.",
    ),
)


def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Vectorised transparent rule score in [0, 1] (max of triggered rules)."""
    score = np.zeros(len(df), dtype=float)
    for rule in RULE_DEFS:
        score = np.maximum(score, np.where(rule.mask(df), rule.weight, 0.0))
    return score


def fired_rules(row: Mapping) -> list[RuleHit]:
    """Rules triggered by a single transaction row (for explanation)."""
    frame = pd.DataFrame([dict(row)])
    hits: list[RuleHit] = []
    for rule in RULE_DEFS:
        if bool(rule.mask(frame)[0]):
            hits.append(RuleHit(rule.code, rule.weight, rule.topic, rule.template(row)))
    return hits
