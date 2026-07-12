"""FX ensemble and fee-aware decision engine (Phase 12).

Combines the per-pair-weighted forecasts of the models that passed the Phase 9/10
backtest into one forecast distribution, then turns it into a convert/hold/split
decision that accounts for the 0.5% conversion fee, model disagreement, forecast
uncertainty, the user's amount, and risk preference. Output matches the frozen
FxAdvisoryResponse contract. FX output is a scenario estimate, never guaranteed profit.
"""

from app.fx.decision.config import DecisionConfig
from app.fx.decision.engine import decide

__all__ = ["DecisionConfig", "decide"]
