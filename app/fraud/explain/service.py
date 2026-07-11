"""Combine triggered rules and SHAP contributions into ranked Indonesian factors.

Precedence: transparent rules carry value-specific wording and are preferred per
topic; SHAP fills in model-only signals. Factors are de-duplicated by topic, ranked,
and capped at ``max_factors`` (3-5). A flagged transaction always returns at least
one factor. Deterministic and LLM-free by construction.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from app.fraud.explain.templates import FALLBACK_FACTOR, FEATURE_TEMPLATE, FEATURE_TOPIC
from app.fraud.rules_engine import fired_rules

DEFAULT_MAX_FACTORS = 5


@dataclass
class Explanation:
    factors: list[str] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)  # {topic, message, source, code, score}


def _rank_candidates(candidates: dict[str, dict], max_factors: int) -> Explanation:
    ordered = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)[:max_factors]
    return Explanation(
        factors=[c["message"] for c in ordered],
        details=[
            {"topic": c["topic"], "message": c["message"], "source": c["source"],
             "code": c["code"], "score": round(c["score"], 4)}
            for c in ordered
        ],
    )


def explain_transaction(
    row: Mapping,
    shap_row: Mapping[str, float] | None = None,
    max_factors: int = DEFAULT_MAX_FACTORS,
    ensure_reason: bool = False,
) -> Explanation:
    """Build ranked factors for one transaction.

    ``row`` must contain the model features plus raw ``hour``/``currency``/
    ``origin_country`` used by rule and template wording. ``shap_row`` maps feature ->
    SHAP contribution (positive = pushes toward fraud); optional. When ``ensure_reason``
    is True and nothing triggers, a generic fallback factor is returned.
    """
    candidates: dict[str, dict] = {}

    # 1) Transparent rules (preferred per topic; carry value-specific wording).
    for hit in fired_rules(row):
        existing = candidates.get(hit.topic)
        if existing is None or hit.weight > existing.get("rule_weight", 0):
            candidates[hit.topic] = {
                "topic": hit.topic, "message": hit.message, "source": "rule",
                "code": hit.code, "rule_weight": hit.weight, "shap_norm": 0.0,
            }

    # 2) SHAP-derived model signals (only contributions pushing toward fraud).
    if shap_row:
        positives = {
            f: v for f, v in shap_row.items()
            if v > 0 and f in FEATURE_TOPIC and f in FEATURE_TEMPLATE
        }
        max_pos = max(positives.values(), default=0.0)
        for feature, value in positives.items():
            topic = FEATURE_TOPIC[feature]
            norm = value / max_pos if max_pos > 0 else 0.0
            existing = candidates.get(topic)
            if existing is not None:
                existing["shap_norm"] = max(existing["shap_norm"], norm)
            else:
                candidates[topic] = {
                    "topic": topic, "message": FEATURE_TEMPLATE[feature](row),
                    "source": "model", "code": feature, "rule_weight": 0.0, "shap_norm": norm,
                }

    # 3) Combined ranking score: rules dominate, model agreement/strength boosts.
    for c in candidates.values():
        if c["rule_weight"] > 0:
            c["score"] = c["rule_weight"] + 0.15 * c["shap_norm"]
        else:
            c["score"] = 0.35 + 0.45 * c["shap_norm"]

    explanation = _rank_candidates(candidates, max_factors)
    if not explanation.factors and ensure_reason:
        explanation.factors = [FALLBACK_FACTOR]
        explanation.details = [{"topic": "model", "message": FALLBACK_FACTOR,
                                "source": "model", "code": "fallback", "score": 0.0}]
    return explanation


def explain_if_flagged(
    risk: float,
    threshold: float,
    row: Mapping,
    shap_row: Mapping[str, float] | None = None,
    max_factors: int = DEFAULT_MAX_FACTORS,
) -> Explanation:
    """Return factors only when the transaction is flagged; empty otherwise.

    Flagged transactions always get at least one reason (Phase 6 definition of done).
    """
    if risk < threshold:
        return Explanation()
    return explain_transaction(row, shap_row, max_factors=max_factors, ensure_reason=True)
