"""SHAP (TreeSHAP) helpers over the CatBoost fraud model.

Uses CatBoost's built-in ShapValues (an exact TreeSHAP implementation), so no extra
runtime dependency is required and values are consistent with the served model.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def shap_matrix(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    """Per-row SHAP contributions, shape (n_rows, n_features).

    The last column returned by CatBoost is the expected value (bias) and is dropped.
    """
    values = model.get_feature_importance(Pool(x), type="ShapValues")
    return np.asarray(values)[:, :-1]


def global_shap_summary(model: CatBoostClassifier, x: pd.DataFrame) -> dict[str, float]:
    """Mean absolute SHAP per feature, sorted descending (global importance)."""
    matrix = shap_matrix(model, x)
    mean_abs = np.abs(matrix).mean(axis=0)
    summary = {feature: round(float(v), 6) for feature, v in zip(x.columns, mean_abs)}
    return dict(sorted(summary.items(), key=lambda kv: kv[1], reverse=True))


def shap_row_dict(shap_row: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Map a single SHAP row to {feature: contribution}."""
    return {name: float(v) for name, v in zip(feature_names, shap_row)}
