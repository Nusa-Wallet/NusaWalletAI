"""Individual fraud model components: CatBoost, Isolation Forest, transparent rules.

Each produces a score in [0, 1] (probability of anomaly) so the ensemble can combine
them uniformly. The rules component is deterministic and human-readable; it is also
the "rules-only" baseline the supervised ensemble must beat.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest


def default_catboost_params(seed: int = 42) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "auto_class_weights": "Balanced",  # handles the ~5% positive rate
        "allow_writing_files": False,
    }


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    seed: int = 42,
    early_stopping_rounds: int = 50,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**(params or default_catboost_params(seed)))
    model.fit(
        x_train,
        y_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model


def catboost_proba(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


@dataclass
class IsolationScorer:
    """Isolation Forest wrapped with a fixed train-time normalisation to [0, 1]."""

    model: IsolationForest
    lo: float
    hi: float

    def score(self, x: pd.DataFrame) -> np.ndarray:
        raw = -self.model.score_samples(x)  # higher = more anomalous
        span = self.hi - self.lo
        if span <= 1e-12:
            return np.zeros(len(x))
        return np.clip((raw - self.lo) / span, 0.0, 1.0)


def train_isolation(x_train: pd.DataFrame, contamination: float = 0.05, seed: int = 42) -> IsolationScorer:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    ).fit(x_train)
    raw = -model.score_samples(x_train)
    lo, hi = float(np.percentile(raw, 1)), float(np.percentile(raw, 99))
    return IsolationScorer(model=model, lo=lo, hi=hi)


# --- Transparent business rules --------------------------------------------

# Each rule contributes a fixed score when triggered; the row score is the strongest
# triggered signal (the conservative policy used by the live demo service).
def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Vectorised transparent rule score in [0, 1] (max of triggered rules)."""
    score = np.zeros(len(df), dtype=float)
    conds = {
        "amount_ratio_user": df["amount_ratio_user"].to_numpy() > 5,
        "amount_zscore_user": df["amount_zscore_user"].to_numpy() > 4,
        "odd_hour": df["hour"].to_numpy() < 6,
        "payer_name_quality": df["payer_name_quality"].to_numpy() < 0.3,
        "payer_velocity_10m": df["payer_velocity_10m"].to_numpy() >= 3,
        "duplicate_similarity": df["duplicate_similarity"].to_numpy() > 0.9,
        "new_payer_high_amount": (~df["payer_seen_before"].to_numpy()) & (df["amount_ratio_user"].to_numpy() > 3),
        "currency_deviation": ~df["currency_seen_before"].to_numpy(),
        "country_deviation": ~df["country_seen_before"].to_numpy(),
    }
    weights = {
        "amount_ratio_user": 0.85, "amount_zscore_user": 0.75, "odd_hour": 0.60,
        "payer_name_quality": 0.75, "payer_velocity_10m": 0.80, "duplicate_similarity": 0.70,
        "new_payer_high_amount": 0.70, "currency_deviation": 0.45, "country_deviation": 0.45,
    }
    for name, mask in conds.items():
        score = np.maximum(score, np.where(mask, weights[name], 0.0))
    return score
