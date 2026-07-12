"""Live FX advisory serving the Phase 12 fee-aware decision engine.

Fetches the recent rate series, produces a fast forecast (statistical drift comparator —
no heavy model load at request time), and runs the decision engine with the caller's
amount, horizon, and risk preference to return the full FxAdvisoryResponse. Falls back to
the legacy statistical advisory if anything fails, so the endpoint always answers.

Serving note: the offline research ensemble (Chronos-2/TimesFM/NHITS) is too heavy to run
per request; the drift forecaster feeding the decision engine keeps the endpoint fast, and
the engine's conservative, fee-aware behaviour is what drives the recommendation.
"""

import numpy as np

from app.fx import service as legacy_service
from app.fx.backtest.adapters import StatisticalComparator
from app.fx.data import fetch_series
from app.fx.features import compute_statistics

_forecaster = StatisticalComparator()


def advise(
    base: str,
    quote: str,
    amount: float | None = None,
    horizon_days: int = 7,
    risk_preference: str = "MODERATE",
) -> dict:
    # Imported here to avoid a heavy import at module load if decision deps change.
    from app.fx.decision import decide

    try:
        series = np.asarray(fetch_series(base, quote, days=max(60, horizon_days * 8)), dtype=float)
        if len(series) < 8 or np.any(series <= 0):
            raise ValueError("insufficient FX series")

        forecast = _forecaster.forecast_batch([series], horizon_days)
        step = horizon_days - 1
        current = float(series[-1])
        forecast_rate = float(forecast.point[0, step])
        lower = max(float(forecast.quantiles[0.1][0, step]), 1e-6)
        upper = float(forecast.quantiles[0.9][0, step])

        decision = decide(
            pair=f"{base}/{quote}", current_rate=current, forecast_rate=forecast_rate,
            forecast_lower=lower, forecast_upper=upper, disagreement=0.0,
            amount=amount, horizon_days=horizon_days, risk_preference=risk_preference,
        )
        stats = compute_statistics(list(series))
        response = decision.to_dict()
        response.update({
            "ma_7d": round(stats.ma_7d, 4),
            "volatility_7d": round(stats.volatility_7d, 4),
            "z_score": round(stats.z_score, 3),
        })
        return response
    except Exception:
        # Legacy statistical advisory keeps the endpoint answering on any failure.
        return legacy_service.advise(base, quote)
