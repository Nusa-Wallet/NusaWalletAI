"""Feature computation shared by training and online fraud inference.

Two layers live here:

* ``legacy_features`` / ``TYPICAL_AMOUNT`` — the minimal inputs of the current demo
  Isolation Forest model (kept for backward compatibility until Phase 7).
* The Phase 4 canonical feature transforms: stateless per-field helpers plus
  :func:`build_features`, which assembles the exact :data:`MODEL_FEATURES` vector from
  a single transaction and its pre-computed historical context. Both the training
  batch pass and future online inference call ``build_features`` so a model sees an
  identical vector in both settings.
"""

from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt

from app.fraud.feature_spec import MODEL_FEATURES

TYPICAL_AMOUNT = {
    "IDR": 5_000_000,
    "USD": 300,
    "SGD": 400,
    "EUR": 280,
    "MYR": 1200,
}


def legacy_features(amount: float, currency: str, hour: int) -> list[float]:
    typical = TYPICAL_AMOUNT[currency.upper()]
    amount_ratio = amount / typical
    odd_hour = 1.0 if hour < 6 else 0.0
    return [amount_ratio, odd_hour]


# --- Phase 4 canonical feature transforms -----------------------------------

# Names that signal a missing/placeholder payer identity.
_PLACEHOLDER_NAMES = frozenset(
    {"test", "n/a", "na", "nan", "unknown", "null", "none", "-", ".", "??"}
)


def hour_sin(hour: int) -> float:
    """Cyclical encoding so hour 23 and hour 0 are adjacent."""
    return round(sin(2 * pi * hour / 24), 6)


def hour_cos(hour: int) -> float:
    return round(cos(2 * pi * hour / 24), 6)


def payer_name_quality(name: str) -> float:
    """Heuristic plausibility of a payer name in ``[0, 1]`` (stateless).

    Empty/too-short/placeholder/repeated-character names score low; a two-token
    mostly-alphabetic name scores high. Deterministic and free of external calls.
    """
    s = (name or "").strip()
    if len(s) < 3:
        return 0.0
    lowered = s.lower()
    if lowered in _PLACEHOLDER_NAMES:
        return 0.1
    if len(set(lowered.replace(" ", ""))) <= 1:  # e.g. "zzz", "aaaa"
        return 0.1
    letters = sum(c.isalpha() for c in s)
    score = letters / len(s)  # penalise digits/symbols
    if len(s.split()) >= 2:
        score = min(1.0, score + 0.1)  # full-name bonus
    else:
        score *= 0.7  # single token is weaker
    return round(max(0.0, min(1.0, score)), 4)


def duplicate_similarity(
    amount_idr: float,
    payer_id: str,
    recent_user_txns: "tuple[tuple[float, str, float], ...]",
) -> float:
    """Similarity in ``[0, 1]`` to the closest of the user's recent past txns.

    ``recent_user_txns`` is ``(amount_idr, payer_id, age_seconds)`` for the same
    user within the trailing 24h window (past only). A near-identical amount from the
    same payer scores ~1.0, catching duplicated/replayed payments.
    """
    best = 0.0
    for amt, pid, _age in recent_user_txns:
        denom = max(abs(amt), 1e-9)
        amount_sim = 1.0 - min(1.0, abs(amount_idr - amt) / denom)
        payer_factor = 1.0 if pid == payer_id else 0.6
        best = max(best, amount_sim * payer_factor)
    return round(best, 4)


@dataclass(frozen=True)
class RawTransaction:
    """The transaction's own fields (no history)."""

    amount_idr: float
    payer_id: str
    payer_name: str
    hour: int
    day_of_week: int


@dataclass(frozen=True)
class HistoricalContext:
    """Past-only aggregates for the transacting user/payer.

    Defaults encode the missing-value policy for a first-ever transaction.
    """

    user_txn_count: int = 0
    user_amount_mean_idr: float = 0.0
    user_amount_std_idr: float = 0.0
    is_new_payer: bool = True
    payer_seen_before: bool = False
    payer_age_days: int = 0
    payer_velocity_10m: int = 0
    payer_velocity_24h: int = 0
    user_velocity_24h: int = 0
    country_seen_before: bool = False
    currency_seen_before: bool = False
    # (amount_idr, payer_id, age_seconds) for the user's trailing-24h transactions.
    recent_user_txns: tuple[tuple[float, str, float], ...] = field(default_factory=tuple)


def build_features(raw: RawTransaction, ctx: HistoricalContext) -> dict[str, float]:
    """Assemble the canonical feature vector (ordered by :data:`MODEL_FEATURES`)."""
    if ctx.user_txn_count > 0 and ctx.user_amount_mean_idr > 0:
        ratio = raw.amount_idr / ctx.user_amount_mean_idr
    else:
        ratio = 1.0
    if ctx.user_txn_count > 0 and ctx.user_amount_std_idr > 1e-9:
        zscore = (raw.amount_idr - ctx.user_amount_mean_idr) / ctx.user_amount_std_idr
    else:
        zscore = 0.0

    values = {
        "amount_idr": round(float(raw.amount_idr), 2),
        "hour_sin": hour_sin(raw.hour),
        "hour_cos": hour_cos(raw.hour),
        "day_of_week": int(raw.day_of_week),
        "amount_ratio_user": round(ratio, 4),
        "amount_zscore_user": round(zscore, 4),
        "duplicate_similarity": duplicate_similarity(
            raw.amount_idr, raw.payer_id, ctx.recent_user_txns
        ),
        "payer_velocity_10m": int(ctx.payer_velocity_10m),
        "payer_velocity_24h": int(ctx.payer_velocity_24h),
        "user_velocity_24h": int(ctx.user_velocity_24h),
        "is_new_payer": bool(ctx.is_new_payer),
        "payer_seen_before": bool(ctx.payer_seen_before),
        "payer_age_days": int(ctx.payer_age_days),
        "payer_name_quality": payer_name_quality(raw.payer_name),
        "country_seen_before": bool(ctx.country_seen_before),
        "currency_seen_before": bool(ctx.currency_seen_before),
    }
    return {name: values[name] for name in MODEL_FEATURES}
