"""Ensemble combination, probability calibration, and threshold selection.

The final risk probability is a weighted blend of the supervised (CatBoost), anomaly
(Isolation Forest), and rules scores, passed through an isotonic calibrator fit on the
validation split so the number reads as a real probability. The decision threshold is
chosen on validation to respect the proposal's false-positive-rate ceiling.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

DEFAULT_WEIGHTS = {"supervised": 0.6, "anomaly": 0.25, "rules": 0.15}


def blend(supervised: np.ndarray, anomaly: np.ndarray, rules: np.ndarray, weights: dict) -> np.ndarray:
    return (
        weights["supervised"] * supervised
        + weights["anomaly"] * anomaly
        + weights["rules"] * rules
    )


@dataclass
class EnsembleModel:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    calibrator: IsotonicRegression | None = None
    threshold: float = 0.5

    def risk(self, supervised: np.ndarray, anomaly: np.ndarray, rules: np.ndarray) -> np.ndarray:
        raw = blend(supervised, anomaly, rules, self.weights)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict(raw)

    def flag(self, risk: np.ndarray) -> np.ndarray:
        return risk >= self.threshold


def fit_calibrator(raw_scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic calibrator mapping blended score -> calibrated probability."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_scores, y)
    return iso


def select_threshold(y: np.ndarray, risk: np.ndarray, max_fpr: float = 0.2) -> float:
    """Pick the threshold maximising F1 among those with validation FPR <= max_fpr.

    Falls back to the plain F1-maximising threshold if none satisfy the FPR ceiling.
    """
    candidates = np.unique(np.round(risk, 4))
    neg = y == 0
    n_neg = max(int(neg.sum()), 1)

    best_t, best_f1 = 0.5, -1.0
    best_t_uncon, best_f1_uncon = 0.5, -1.0
    for t in candidates:
        pred = risk >= t
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1_uncon:
            best_f1_uncon, best_t_uncon = f1, float(t)
        fpr = float((pred & neg).sum()) / n_neg
        if fpr <= max_fpr and f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t if best_f1 >= 0 else best_t_uncon
