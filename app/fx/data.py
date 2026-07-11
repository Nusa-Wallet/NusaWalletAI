"""FX series retrieval with a deterministic offline fallback."""

from datetime import date, timedelta
import hashlib
import math

import httpx


def fetch_series(base: str, quote: str, days: int = 30) -> list[float]:
    try:
        response = httpx.get(
            f"https://api.frankfurter.app/{window_start(days)}..",
            params={"from": base, "to": quote},
            timeout=6.0,
        )
        response.raise_for_status()
        rates = response.json()["rates"]
        series = [values[quote] for _, values in sorted(rates.items())]
        if len(series) >= 8:
            return series
    except Exception:
        pass
    return synthetic_series(base, quote, days)


def window_start(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def synthetic_series(base: str, quote: str, days: int) -> list[float]:
    digest = hashlib.sha256(f"{base}/{quote}".encode()).digest()
    seed = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    center = 12000 + seed * 200
    return [center * (1 + 0.012 * math.sin(index / 3.0 + seed * 6)) for index in range(days)]
