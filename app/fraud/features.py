"""Feature computation shared by training and online fraud inference."""

TYPICAL_AMOUNT = {
    "IDR": 5_000_000,
    "USD": 300,
    "SGD": 400,
    "EUR": 280,
    "MYR": 1200,
}


def legacy_features(amount: float, currency: str, hour: int) -> list[float]:
    typical = TYPICAL_AMOUNT[currency.upper()]
    amount_ratio = amount / typical
    odd_hour = 1.0 if hour < 6 else 0.0
    return [amount_ratio, odd_hour]
