"""Configuration for the FX ensemble and decision engine."""

from dataclasses import dataclass, field

MODEL_VERSION = "fx-decision-1.0.0"

# Models combined into the ensemble (those that passed the Phase 9/10 backtest).
# statistical-drift is included as the decision-strong comparator.
DEFAULT_MODELS = ("chronos-2", "timesfm-2.5", "nhits-global", "statistical-drift")

# Risk preference -> multiplier on the fraction held back (waiting to convert).
RISK_HOLD_MULTIPLIER = {"CONSERVATIVE": 0.5, "MODERATE": 1.0, "AGGRESSIVE": 1.5}


@dataclass(frozen=True)
class DecisionConfig:
    fee_rate: float = 0.005
    # Confidence: uncertainty grows with forecast interval width and model disagreement
    # (both relative to the current rate). Scales are the level that saturates each term.
    interval_scale: float = 0.04
    disagree_scale: float = 0.02
    weight_interval: float = 0.6
    weight_disagree: float = 0.4
    # Action thresholds on "percent to convert now".
    convert_now_pct: int = 90
    hold_pct: int = 10
    min_confidence: float = 0.05
    max_confidence: float = 0.99
    models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODELS)

    def risk_multiplier(self, risk_preference: str) -> float:
        return RISK_HOLD_MULTIPLIER.get(risk_preference.upper(), 1.0)
