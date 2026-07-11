"""Past-only historical/behavioural feature computation.

A single forward pass over chronologically ordered transactions. Every feature for
row *i* is derived exclusively from rows strictly before *i* in time, so there is no
future or label leakage. Training and inference should reuse this same logic.

Missing-value policy: for a user's first-ever transaction there is no history, so
``amount_ratio_user`` defaults to 1.0 and ``amount_zscore_user`` to 0.0; all
"seen_before" flags are False and velocities are 0.
"""

from bisect import bisect_left
from math import sqrt

import numpy as np
import pandas as pd

WINDOW_10M = 600.0
WINDOW_24H = 86_400.0

HISTORICAL_FEATURE_COLUMNS = (
    "hour",
    "day_of_week",
    "is_new_payer",
    "payer_seen_before",
    "payer_age_days",
    "country_seen_before",
    "currency_seen_before",
    "payer_velocity_10m",
    "payer_velocity_24h",
    "user_velocity_24h",
    "amount_ratio_user",
    "amount_zscore_user",
)


def _count_since(times: list[float], cutoff: float) -> int:
    """Number of past timestamps in ``times`` that are >= ``cutoff`` (times sorted)."""
    return len(times) - bisect_left(times, cutoff)


def compute_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to ``df`` (assumed chronologically sorted)."""
    n = len(df)
    user_ids = df["user_id"].to_numpy()
    payer_ids = df["payer_id"].to_numpy()
    amount_idr = df["amount_idr"].to_numpy(dtype=float)
    currencies = df["currency"].to_numpy()
    countries = df["origin_country"].to_numpy()
    occurred = df["occurred_at"]
    ts = occurred.astype("int64").to_numpy() / 1e9  # epoch seconds

    hour = occurred.dt.hour.to_numpy()
    day_of_week = occurred.dt.dayofweek.to_numpy()

    is_new_payer = np.empty(n, dtype=bool)
    payer_seen_before = np.empty(n, dtype=bool)
    payer_age_days = np.empty(n, dtype=np.int64)
    country_seen_before = np.empty(n, dtype=bool)
    currency_seen_before = np.empty(n, dtype=bool)
    payer_velocity_10m = np.empty(n, dtype=np.int64)
    payer_velocity_24h = np.empty(n, dtype=np.int64)
    user_velocity_24h = np.empty(n, dtype=np.int64)
    amount_ratio_user = np.empty(n, dtype=float)
    amount_zscore_user = np.empty(n, dtype=float)

    user_times: dict[int, list[float]] = {}
    payer_times: dict[str, list[float]] = {}
    payer_first_seen: dict[str, float] = {}
    user_countries: dict[int, set] = {}
    user_currencies: dict[int, set] = {}
    user_payers: dict[int, set] = {}
    user_stats: dict[int, list[float]] = {}  # [count, mean, M2]

    for i in range(n):
        uid = int(user_ids[i])
        pid = payer_ids[i]
        t = float(ts[i])
        amt = float(amount_idr[i])

        ptimes = payer_times.get(pid)
        if ptimes is None:
            is_new_to_dataset = True
            first_seen = t
        else:
            is_new_to_dataset = False
            first_seen = payer_first_seen[pid]
        payer_age_days[i] = int((t - first_seen) // WINDOW_24H)

        seen_payers = user_payers.get(uid, ())
        seen_before = pid in seen_payers
        payer_seen_before[i] = seen_before
        is_new_payer[i] = not seen_before

        payer_velocity_10m[i] = _count_since(ptimes, t - WINDOW_10M) if ptimes else 0
        payer_velocity_24h[i] = _count_since(ptimes, t - WINDOW_24H) if ptimes else 0
        utimes = user_times.get(uid)
        user_velocity_24h[i] = _count_since(utimes, t - WINDOW_24H) if utimes else 0

        country_seen_before[i] = countries[i] in user_countries.get(uid, ())
        currency_seen_before[i] = currencies[i] in user_currencies.get(uid, ())

        stats = user_stats.get(uid)
        if stats is None or stats[0] == 0:
            amount_ratio_user[i] = 1.0
            amount_zscore_user[i] = 0.0
        else:
            cnt, mean, m2 = stats
            std = sqrt(m2 / cnt) if cnt > 0 else 0.0
            amount_ratio_user[i] = amt / mean if mean > 0 else 1.0
            amount_zscore_user[i] = (amt - mean) / std if std > 1e-9 else 0.0

        # --- update state AFTER computing (keeps everything past-only) ---
        if ptimes is None:
            payer_times[pid] = [t]
            payer_first_seen[pid] = first_seen
        else:
            ptimes.append(t)
        user_times.setdefault(uid, []).append(t)
        user_countries.setdefault(uid, set()).add(countries[i])
        user_currencies.setdefault(uid, set()).add(currencies[i])
        user_payers.setdefault(uid, set()).add(pid)
        s = user_stats.setdefault(uid, [0.0, 0.0, 0.0])
        s[0] += 1
        delta = amt - s[1]
        s[1] += delta / s[0]
        s[2] += delta * (amt - s[1])
        _ = is_new_to_dataset  # reserved for future dataset-level payer features

    return pd.DataFrame(
        {
            "hour": hour.astype(np.int64),
            "day_of_week": day_of_week.astype(np.int64),
            "is_new_payer": is_new_payer,
            "payer_seen_before": payer_seen_before,
            "payer_age_days": payer_age_days,
            "country_seen_before": country_seen_before,
            "currency_seen_before": currency_seen_before,
            "payer_velocity_10m": payer_velocity_10m,
            "payer_velocity_24h": payer_velocity_24h,
            "user_velocity_24h": user_velocity_24h,
            "amount_ratio_user": np.round(amount_ratio_user, 4),
            "amount_zscore_user": np.round(amount_zscore_user, 4),
        },
        index=df.index,
    )
