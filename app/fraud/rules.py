"""Transparent fraud rules and their user-facing explanations."""

from dataclasses import dataclass

from app.fraud.features import TYPICAL_AMOUNT


@dataclass(frozen=True)
class RuleResult:
    score: float
    factors: list[str]


def evaluate_rules(amount: float, currency: str, payer_name: str, hour: int) -> RuleResult:
    factors: list[str] = []
    score = 0.0
    typical = TYPICAL_AMOUNT[currency.upper()]
    if amount > typical * 5:
        factors.append(f"Nominal {amount} jauh di atas rata-rata {currency} (~{typical}).")
        score = max(score, 0.8)
    if hour < 6:
        factors.append("Transaksi terjadi pada jam tidak biasa (00:00-06:00).")
        score = max(score, 0.6)
    if not payer_name or len(payer_name.strip()) < 3:
        factors.append("Nama pembayar tidak lengkap / mencurigakan.")
        score = max(score, 0.7)
    return RuleResult(score=score, factors=factors)
