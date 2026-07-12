"""Per-pair ensemble weights from validation forecast error.

Weights are inverse validation pinball loss (averaged over horizons), normalised per
pair, so the more accurate model on a pair gets more weight — the plan's "bobot per pair
berdasarkan validation error". Missing model/pair entries are skipped gracefully.
"""

import numpy as np


def _model_key(model: str, pair: str, split: str, horizon: int) -> str:
    return f"{model}|{pair}|{split}|h{horizon}"


def derive_weights(
    per_model_metrics: dict[str, dict],
    models: tuple[str, ...],
    pairs: list[str],
    horizons: tuple[int, ...] = (1, 3, 7),
    metric: str = "mean_pinball",
    split: str = "val",
) -> dict[str, dict[str, float]]:
    """Return {pair: {model: weight}} with weights summing to 1 per pair."""
    weights: dict[str, dict[str, float]] = {}
    for pair in pairs:
        inverse: dict[str, float] = {}
        for model in models:
            metrics = per_model_metrics.get(model, {})
            errors = [
                metrics[_model_key(model, pair, split, h)][metric]
                for h in horizons
                if _model_key(model, pair, split, h) in metrics
            ]
            if not errors:
                continue
            mean_error = float(np.mean(errors))
            if mean_error > 0:
                inverse[model] = 1.0 / mean_error
        total = sum(inverse.values())
        if total > 0:
            weights[pair] = {m: round(w / total, 6) for m, w in inverse.items()}
        else:
            equal = 1.0 / len(models)
            weights[pair] = {m: equal for m in models}
    return weights
