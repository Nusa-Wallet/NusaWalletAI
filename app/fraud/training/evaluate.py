"""Evaluation metrics for imbalanced fraud detection.

Reports the metrics the proposal requires: precision/recall/F1, PR-AUC, ROC-AUC,
false-positive rate, Brier score (calibration), and recall per anomaly type. Accuracy
is deliberately omitted as the headline number because it is misleading at ~5% positives.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, risk: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (risk >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # AUC metrics need both classes present.
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(average_precision_score(y_true, risk)), 4) if both else None,
        "roc_auc": round(float(roc_auc_score(y_true, risk)), 4) if both else None,
        "fpr": round(float(fpr), 4),
        "brier": round(float(brier_score_loss(y_true, np.clip(risk, 0, 1))), 4),
        "support_positive": int(tp + fn),
        "support_negative": int(tn + fp),
    }


def recall_per_anomaly_type(df_test: pd.DataFrame, flagged: np.ndarray) -> dict:
    """Fraction of each anomaly type that was flagged (recall by scenario)."""
    out: dict[str, float] = {}
    types = df_test["anomaly_type"].to_numpy()
    for t in sorted(set(types)):
        mask = types == t
        if mask.sum() == 0:
            continue
        out[str(t)] = round(float(flagged[mask].mean()), 4)
    return out
