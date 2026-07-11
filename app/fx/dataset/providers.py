"""Rate retrieval: real ECB-via-Frankfurter fetch and a deterministic fallback.

``fetch_eur_base`` returns a DataFrame of EUR-based rates (units of each currency per
1 EUR) indexed by observation date, plus a provenance dict. Cross-rates are computed
locally from this single source. A synthetic generator provides an offline fallback so
the build and tests never require network access; provenance always records which
provider produced the data.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.fx.dataset.config import (
    FRANKFURTER_BASE_URL,
    PROVIDER_FALLBACK,
    PROVIDER_REAL,
    FxDatasetConfig,
)


def fetch_eur_base(config: FxDatasetConfig, timeout: float = 60.0) -> tuple[pd.DataFrame, dict]:
    """Fetch EUR-based daily rates for the universe. Falls back to synthetic offline."""
    symbols = config.symbols()
    url = f"{FRANKFURTER_BASE_URL}/{config.start_date}..{config.end()}"
    try:
        import httpx

        response = httpx.get(
            url,
            params={"base": config.base, "symbols": ",".join(symbols)},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        frame = _eur_frame_from_payload(payload, config)
        provenance = {
            "provider": PROVIDER_REAL,
            "source_url": str(response.url),
            "base": config.base,
            "symbols": symbols,
            "start_date": config.start_date,
            "end_date": config.end(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_observations": int(len(frame)),
        }
        return frame, provenance
    except Exception as exc:  # offline / API change -> deterministic fallback
        frame = synthetic_eur_base(config)
        provenance = {
            "provider": PROVIDER_FALLBACK,
            "reason": f"{type(exc).__name__}: {exc}",
            "base": config.base,
            "symbols": symbols,
            "start_date": config.start_date,
            "end_date": config.end(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_observations": int(len(frame)),
            "seed": config.seed,
        }
        return frame, provenance


def _eur_frame_from_payload(payload: dict, config: FxDatasetConfig) -> pd.DataFrame:
    rates = payload["rates"]  # {date: {ccy: rate}}
    frame = pd.DataFrame(rates).T
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame[config.base] = 1.0  # base per base
    return frame[[c for c in config.currencies if c in frame.columns]]


def synthetic_eur_base(config: FxDatasetConfig) -> pd.DataFrame:
    """Deterministic EUR-based rates over business days (offline fallback)."""
    dates = pd.bdate_range(config.start_date, config.end(), tz="UTC")
    rng = np.random.default_rng(config.seed)
    data = {config.base: np.ones(len(dates))}
    # Rough starting levels (units per EUR) so magnitudes are plausible.
    anchors = {"USD": 1.1, "JPY": 130.0, "GBP": 0.85, "SGD": 1.5, "MYR": 4.5,
               "IDR": 16000.0, "THB": 38.0, "PHP": 60.0, "INR": 90.0}
    for ccy in config.currencies:
        if ccy == config.base:
            continue
        start = anchors.get(ccy, 10.0)
        # geometric random walk with tiny drift
        steps = rng.normal(0.0, 0.006, size=len(dates))
        series = start * np.exp(np.cumsum(steps))
        data[ccy] = series
    return pd.DataFrame(data, index=dates)[list(config.currencies)]


def verify_cross_rates(
    config: FxDatasetConfig, eur_frame: pd.DataFrame, sample: int = 6, timeout: float = 30.0
) -> dict:
    """Cross-check computed primary cross-rates against direct Frankfurter fetches.

    Best-effort: returns per-pair max relative error, or a skip reason when offline.
    """
    try:
        import httpx

        results = {}
        sample_dates = list(eur_frame.index[-sample:])
        for pair in config.primary_pairs:
            base, quote = pair.split("/")
            computed = (eur_frame[quote] / eur_frame[base]).loc[sample_dates]
            start = sample_dates[0].date().isoformat()
            end = sample_dates[-1].date().isoformat()
            r = httpx.get(
                f"{FRANKFURTER_BASE_URL}/{start}..{end}",
                params={"base": base, "symbols": quote},
                timeout=timeout,
                follow_redirects=True,
            )
            r.raise_for_status()
            direct = pd.Series({pd.to_datetime(d, utc=True): v[quote] for d, v in r.json()["rates"].items()})
            common = computed.index.intersection(direct.index)
            if len(common):
                rel = ((computed.loc[common] - direct.loc[common]).abs() / direct.loc[common]).max()
                results[pair] = round(float(rel), 6)
        return {"method": "direct-frankfurter", "max_rel_error": results}
    except Exception as exc:
        return {"method": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
