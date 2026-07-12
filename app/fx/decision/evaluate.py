"""Evaluate the decision engine's fee-aware outcomes over the ensemble predictions.

For each case the engine's recommended split (convert now vs hold to horizon) is scored:
part is converted now at the current rate, the rest at the realised future rate, both net
of the fee. Reported against immediate full conversion and a perfect-timing oracle.
"""

import numpy as np
import pandas as pd

from app.fx.decision.config import DecisionConfig
from app.fx.decision.engine import decide


def evaluate_decisions(
    ensemble_df: pd.DataFrame,
    config: DecisionConfig | None = None,
    amount: float = 1000.0,
    risk_preference: str = "MODERATE",
) -> pd.DataFrame:
    config = config or DecisionConfig()
    keep = 1 - config.fee_rate
    rows = []
    for r in ensemble_df.itertuples():
        d = decide(
            pair=r.pair, current_rate=r.current_rate, forecast_rate=r.point_forecast,
            forecast_lower=r.q10, forecast_upper=r.q90, disagreement=r.disagreement,
            amount=amount, horizon_days=int(r.horizon), risk_preference=risk_preference,
            config=config,
        )
        convert_now = d.recommended_convert_percentage / 100
        chosen = amount * (convert_now * r.current_rate + (1 - convert_now) * r.actual_rate) * keep
        immediate = amount * r.current_rate * keep
        oracle = amount * max(r.current_rate, r.actual_rate) * keep
        rows.append({
            "pair": r.pair, "evaluation_split": r.evaluation_split, "horizon": r.horizon,
            "action": d.action, "confidence": d.confidence,
            "convert_now_pct": d.recommended_convert_percentage,
            "net_gain_vs_immediate": chosen - immediate, "regret": oracle - chosen,
        })
    return pd.DataFrame(rows)


def summarize(decisions: pd.DataFrame) -> dict:
    out: dict = {"by_split": {}, "by_pair": {}}
    for split, group in decisions.groupby("evaluation_split"):
        out["by_split"][str(split)] = _summary(group)
    for (pair, split), group in decisions.groupby(["pair", "evaluation_split"]):
        out["by_pair"].setdefault(str(split), {})[str(pair)] = _summary(group)
    return out


def _summary(group: pd.DataFrame) -> dict:
    actions = group["action"].value_counts(normalize=True).round(3).to_dict()
    return {
        "n": int(len(group)),
        "mean_net_gain": round(float(group["net_gain_vs_immediate"].mean()), 2),
        "total_net_gain": round(float(group["net_gain_vs_immediate"].sum()), 2),
        "mean_regret": round(float(group["regret"].mean()), 2),
        "max_regret": round(float(group["regret"].max()), 2),
        "mean_confidence": round(float(group["confidence"].mean()), 3),
        "action_mix": {str(k): float(v) for k, v in actions.items()},
    }


def model_net_gain(per_model: dict[str, pd.DataFrame], split: str = "val") -> dict:
    """Total net gain per model from their own backtest decisions (for comparison)."""
    out = {}
    for model, df in per_model.items():
        sub = df[df["evaluation_split"] == split]
        out[model] = round(float(sub["net_gain_vs_immediate"].sum()), 2) if len(sub) else 0.0
    return out
