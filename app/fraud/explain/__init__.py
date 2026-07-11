"""Fraud explainability (Phase 6).

Produces 3-5 ranked, human-readable Indonesian factors for a high-risk transaction by
combining triggered transparent rules with the CatBoost model's SHAP contributions.
No LLM is involved in choosing the reasons — every factor is deterministic and tied to
an actual feature value or a triggered rule.
"""

from app.fraud.explain.service import Explanation, explain_if_flagged, explain_transaction

__all__ = ["Explanation", "explain_transaction", "explain_if_flagged", "default_factor"]


def default_factor() -> str:
    """Neutral factor for the legacy demo service when no anomaly is significant."""
    return "Tidak ada anomali signifikan terdeteksi."
