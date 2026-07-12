"""Fee-aware FX decision engine.

Given an ensemble forecast (point + interval), model disagreement, the user's amount,
and risk preference, produce a convert/hold/split decision. The rate is quote-per-base
(e.g. IDR per SGD), so a forecast above the current rate means the foreign currency is
expected to appreciate — a reason to hold part of the conversion. The 0.5% fee is the
hurdle the expected move must clear before holding is worthwhile.
"""

from dataclasses import dataclass

import numpy as np

from app.fx.decision.config import MODEL_VERSION, DecisionConfig


@dataclass(frozen=True)
class Decision:
    pair: str
    action: str
    confidence: float
    current_rate: float
    forecast_rate: float
    forecast_lower: float
    forecast_upper: float
    recommended_convert_percentage: int
    estimated_gain_loss: float | None
    scenario_best: float
    scenario_worst: float
    rationale: str
    reasons: list[str]
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _confidence(interval_norm: float, disagree_norm: float, config: DecisionConfig) -> float:
    uncertainty = (
        config.weight_interval * interval_norm / config.interval_scale
        + config.weight_disagree * disagree_norm / config.disagree_scale
    )
    uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
    return round(float(np.clip(1.0 - uncertainty, config.min_confidence, config.max_confidence)), 2)


def decide(
    *,
    pair: str,
    current_rate: float,
    forecast_rate: float,
    forecast_lower: float,
    forecast_upper: float,
    disagreement: float = 0.0,
    amount: float | None = None,
    horizon_days: int = 7,
    risk_preference: str = "MODERATE",
    config: DecisionConfig | None = None,
) -> Decision:
    config = config or DecisionConfig()
    current = float(current_rate)
    expected_return = (forecast_rate - current) / current if current else 0.0
    interval_norm = max(0.0, (forecast_upper - forecast_lower) / current) if current else 0.0
    disagree_norm = max(0.0, disagreement / current) if current else 0.0

    confidence = _confidence(interval_norm, disagree_norm, config)

    # The fee is paid on any conversion, so it is neutral to now-vs-later timing; the
    # timing signal is the expected move itself, scaled so a fee-sized move (0.5%) is
    # "small" and holding grows with larger moves. Confidence gates it: uncertain or
    # disagreeing forecasts hold less. Fee still enters the gain estimate and the scale.
    appreciation = float(np.clip(expected_return / (2 * config.fee_rate), -1.0, 1.0))
    risk_mult = config.risk_multiplier(risk_preference)
    hold_fraction = float(np.clip(confidence * max(0.0, appreciation) * risk_mult, 0.0, 1.0))
    convert_now_pct = int(round(100 * (1 - hold_fraction)))

    if convert_now_pct >= config.convert_now_pct:
        action, convert_now_pct = "CONVERT_NOW", 100
    elif convert_now_pct <= config.hold_pct:
        action, convert_now_pct = "HOLD_TEMPORARILY", 0
    else:
        action = "SPLIT_CONVERSION"

    held_fraction = 1 - convert_now_pct / 100
    # Expected extra IDR from holding the retained part to the horizon, net of the fee.
    estimated_gain_loss = (
        round(float(amount) * held_fraction * (forecast_rate - current) * (1 - config.fee_rate), 2)
        if amount is not None else None
    )

    rationale, reasons = _explain(
        pair, action, expected_return, confidence, convert_now_pct, horizon_days,
        disagree_norm, interval_norm, config,
    )
    return Decision(
        pair=pair, action=action, confidence=confidence,
        current_rate=round(current, 4), forecast_rate=round(float(forecast_rate), 4),
        forecast_lower=round(float(forecast_lower), 4), forecast_upper=round(float(forecast_upper), 4),
        recommended_convert_percentage=convert_now_pct, estimated_gain_loss=estimated_gain_loss,
        scenario_best=round(float(forecast_upper), 4), scenario_worst=round(float(forecast_lower), 4),
        rationale=rationale, reasons=reasons,
    )


def _explain(pair, action, expected_return, confidence, convert_now_pct, horizon,
             disagree_norm, interval_norm, config):
    pct = abs(expected_return) * 100
    conf_pct = round(confidence * 100)
    if action == "CONVERT_NOW":
        if expected_return < 0:
            rationale = (f"Model memperkirakan {pair} melemah ~{pct:.1f}% dalam {horizon} hari; "
                         f"disarankan konversi sekarang untuk menghindari penurunan nilai.")
        else:
            rationale = (f"Perkiraan kenaikan {pair} tidak melebihi biaya konversi {config.fee_rate:.1%}; "
                         f"konversi sekarang lebih menguntungkan.")
    elif action == "HOLD_TEMPORARILY":
        rationale = (f"Model memperkirakan {pair} menguat ~{pct:.1f}% dalam {horizon} hari dengan "
                     f"keyakinan {conf_pct}%; disarankan menahan konversi sementara.")
    else:
        rationale = (f"Ketidakpastian model cukup tinggi (keyakinan {conf_pct}%); disarankan konversi "
                     f"bertahap — {convert_now_pct}% sekarang, sisanya ditahan.")

    reasons = [
        f"Estimasi pergerakan {pair}: {expected_return * 100:+.1f}% dalam {horizon} hari.",
        f"Keyakinan {conf_pct}% (turun saat model tidak sepakat / rentang prediksi lebar).",
    ]
    if disagree_norm > config.disagree_scale:
        reasons.append("Model saling tidak sepakat pada arah pergerakan.")
    if interval_norm > config.interval_scale:
        reasons.append("Rentang prediksi lebar menandakan volatilitas tinggi.")
    reasons.append(f"Keputusan memperhitungkan biaya konversi {config.fee_rate:.1%}.")
    return rationale, reasons
