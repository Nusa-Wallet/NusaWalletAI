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

# Canonical transparent rules live in one place, shared with the explainability
# layer so a triggered rule always has a matching human-readable reason.
from app.fraud.rules_engine import rules_score  # noqa: F401  (re-exported)


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


# Transparent business rules (rules_score) are defined in app.fraud.rules_engine and
# re-exported above so training and explanation share one source of truth.
