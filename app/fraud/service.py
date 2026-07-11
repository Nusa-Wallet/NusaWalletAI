"""Application service orchestrating fraud features, model, rules, and response."""

from app.config import FRAUD_MODEL_VERSION
from app.fraud.ensemble import combine_demo_scores
from app.fraud.explain import default_factor
from app.fraud.model import model
from app.fraud.rules import evaluate_rules


def warm_up() -> None:
    model.train_demo_model()


def score(amount: float, currency: str, payer_name: str, hour: int = 12) -> dict:
    anomaly_score = model.risk_score(amount, currency, hour)
    rules = evaluate_rules(amount, currency, payer_name, hour)
    risk = combine_demo_scores(anomaly_score, rules.score)
    factors = rules.factors or [default_factor()]
    if risk >= 0.7:
        level, action = "HIGH", "REVIEW_REQUIRED"
    elif risk >= 0.3:
        level, action = "MEDIUM", "REVIEW_IF_NEEDED"
    else:
        level, action = "LOW", "ALLOW"
    return {
        "risk_score": risk,
        "risk_level": level,
        "flagged": risk >= 0.7,
        "recommended_action": action,
        "factors": factors,
        "component_scores": {
            "supervised": None,
            "anomaly": anomaly_score,
            "rules": rules.score,
        },
        "model": "IsolationForest + rules",
        "model_version": FRAUD_MODEL_VERSION,
    }
