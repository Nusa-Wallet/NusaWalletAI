"""Fraud / anomaly scoring.

Combines an unsupervised anomaly model (IsolationForest, trained at startup on
synthetic rule-based transactions) with transparent business rules. The final
response lists the contributing factors so the decision is explainable, and the
heavy lifting stays server-side (proposal section 9).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

# Rough per-currency "typical" transaction magnitude, used to normalise amount.
_TYPICAL = {"IDR": 5_000_000, "USD": 300, "SGD": 400, "EUR": 280, "MYR": 1200}

_model: IsolationForest | None = None


def _features(amount: float, currency: str, hour: int) -> list[float]:
    typical = _TYPICAL.get(currency.upper(), 300)
    ratio = amount / typical
    # off-hours flag (00:00-05:00 considered unusual)
    odd_hour = 1.0 if hour < 6 else 0.0
    return [ratio, odd_hour]


def train() -> None:
    """Train on synthetic 'normal' transactions so anomalies stand out."""
    global _model
    rng = np.random.default_rng(42)
    normal_ratio = rng.lognormal(mean=0.0, sigma=0.4, size=2000)  # mostly near 1x
    normal_hour = rng.integers(6, 24, size=2000)
    X = np.column_stack([normal_ratio, (normal_hour < 6).astype(float)])
    _model = IsolationForest(contamination=0.05, random_state=42).fit(X)


def score(amount: float, currency: str, payer_name: str, hour: int = 12) -> dict:
    if _model is None:
        train()

    feats = np.array([_features(amount, currency, hour)])
    # decision_function: higher = more normal. Map to 0..1 risk.
    raw = float(_model.decision_function(feats)[0])
    risk = round(float(np.clip(0.5 - raw, 0.0, 1.0)), 3)

    factors: list[str] = []
    typical = _TYPICAL.get(currency.upper(), 300)
    if amount > typical * 5:
        factors.append(f"Nominal {amount} jauh di atas rata-rata {currency} (~{typical}).")
        risk = max(risk, 0.8)
    if hour < 6:
        factors.append("Transaksi terjadi pada jam tidak biasa (00:00-06:00).")
        risk = max(risk, 0.6)
    if not payer_name or len(payer_name.strip()) < 3:
        factors.append("Nama pembayar tidak lengkap / mencurigakan.")
        risk = max(risk, 0.7)
    if not factors:
        factors.append("Tidak ada anomali signifikan terdeteksi.")

    return {
        "risk_score": round(risk, 3),
        "flagged": risk >= 0.7,
        "factors": factors,
        "model": "IsolationForest + rules",
    }
