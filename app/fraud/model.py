"""Current anomaly model boundary; replaceable by persisted artifacts later."""

import numpy as np
from sklearn.ensemble import IsolationForest

from app.fraud.features import legacy_features


class IsolationFraudModel:
    def __init__(self) -> None:
        self._model: IsolationForest | None = None

    def train_demo_model(self) -> None:
        rng = np.random.default_rng(42)
        normal_ratio = rng.lognormal(mean=0.0, sigma=0.4, size=2000)
        normal_hour = rng.integers(6, 24, size=2000)
        features = np.column_stack([normal_ratio, (normal_hour < 6).astype(float)])
        self._model = IsolationForest(contamination=0.05, random_state=42).fit(features)

    def risk_score(self, amount: float, currency: str, hour: int) -> float:
        if self._model is None:
            self.train_demo_model()
        features = np.array([legacy_features(amount, currency, hour)])
        raw = float(self._model.decision_function(features)[0])
        return round(float(np.clip(0.5 - raw, 0.0, 1.0)), 3)


model = IsolationFraudModel()
