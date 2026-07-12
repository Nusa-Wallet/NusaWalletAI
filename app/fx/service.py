"""Application service for the currently implemented statistical advisory."""

from app.config import FX_MODEL_VERSION
from app.fx.data import fetch_series
from app.fx.decision.statistical import decide
from app.fx.features import compute_statistics


def advise(base: str, quote: str) -> dict:
    stats = compute_statistics(fetch_series(base, quote))
    decision = decide(base, quote, stats)
    confidence = round(min(0.99, 0.5 + abs(stats.z_score) * 0.2), 2)
    return {
        "pair": f"{base}/{quote}",
        "action": decision.action,
        "confidence": confidence,
        "current_rate": round(stats.current, 4),
        "ma_7d": round(stats.ma_7d, 4),
        "volatility_7d": round(stats.volatility_7d, 4),
        "z_score": round(stats.z_score, 3),
        "forecast_rate": None,
        "forecast_lower": None,
        "forecast_upper": None,
        "recommended_convert_percentage": None,
        "estimated_gain_loss": None,
        "scenario_best": round(stats.current + stats.volatility_7d, 4),
        "scenario_worst": round(stats.current - stats.volatility_7d, 4),
        "rationale": decision.rationale,
        "reasons": [],
        "model_version": FX_MODEL_VERSION,
    }
