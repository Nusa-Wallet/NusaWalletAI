"""Score combination policy for current and future fraud components."""


def combine_demo_scores(anomaly_score: float, rule_score: float) -> float:
    # Preserve the legacy conservative policy until calibrated weights exist.
    return round(max(anomaly_score, rule_score), 3)
