"""Validated forecast models will be combined through this boundary."""


def weighted_forecast(forecasts: list[float], weights: list[float]) -> float:
    if not forecasts or len(forecasts) != len(weights):
        raise ValueError("Forecasts and weights must be non-empty and have equal length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Forecast weights must have a positive sum")
    return sum(value * weight for value, weight in zip(forecasts, weights)) / total
