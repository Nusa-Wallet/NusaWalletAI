"""Past-only historical/behavioural feature computation (training batch pass).

A single forward pass over chronologically ordered transactions. For every row it
builds the past-only :class:`HistoricalContext` from streaming per-user/per-payer
state and then calls the shared :func:`app.fraud.features.build_features`, so the
training vectors are produced by the exact same code the online path will use.

No feature for row *i* uses any row at or after *i* in time (state is updated only
after the row is emitted), so there is no future or label leakage. See
``app.fraud.feature_spec`` for the missing-value policy applied to first-ever rows.
"""

from bisect import bisect_left
from collections import deque
from math import sqrt

import pandas as pd

from app.fraud.feature_spec import MODEL_FEATURES
from app.fraud.features import HistoricalContext, RawTransaction, build_features

WINDOW_10M = 600.0
WINDOW_24H = 86_400.0

# Columns emitted by the batch pass: raw ``hour`` for EDA plus every model feature
# except ``amount_idr`` (already present as a raw column on the frame).
FEATURE_FRAME_COLUMNS = ("hour",) + tuple(f for f in MODEL_FEATURES if f != "amount_idr")


def _count_since(times: list[float], cutoff: float) -> int:
    """Number of past timestamps in ``times`` that are >= ``cutoff`` (times sorted)."""
    return len(times) - bisect_left(times, cutoff)


def compute_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to ``df`` (assumed chronologically sorted)."""
    user_ids = df["user_id"].to_numpy()
    payer_ids = df["payer_id"].to_numpy()
    amount_idr = df["amount_idr"].to_numpy(dtype=float)
    payer_names = df["payer_name"].to_numpy()
    currencies = df["currency"].to_numpy()
    countries = df["origin_country"].to_numpy()
    occurred = df["occurred_at"]
    ts = occurred.astype("int64").to_numpy() / 1e9  # epoch seconds
    hours = occurred.dt.hour.to_numpy()
    days = occurred.dt.dayofweek.to_numpy()

    payer_times: dict[str, list[float]] = {}
    payer_first_seen: dict[str, float] = {}
    user_recent: dict[int, deque] = {}          # (t, amount_idr, payer_id) within 24h
    user_countries: dict[int, set] = {}
    user_currencies: dict[int, set] = {}
    user_payers: dict[int, set] = {}
    user_stats: dict[int, list[float]] = {}      # [count, mean, M2]

    rows: list[dict] = []
    for i in range(len(df)):
        uid = int(user_ids[i])
        pid = payer_ids[i]
        t = float(ts[i])
        amt = float(amount_idr[i])

        ptimes = payer_times.get(pid)
        first_seen = payer_first_seen.get(pid, t)

        recent = user_recent.get(uid)
        if recent is not None:
            while recent and (t - recent[0][0]) > WINDOW_24H:
                recent.popleft()

        stats = user_stats.get(uid)
        cnt = stats[0] if stats else 0
        mean = stats[1] if stats else 0.0
        std = sqrt(stats[2] / cnt) if stats and cnt > 0 else 0.0

        seen_payers = user_payers.get(uid, ())
        recent_tuples = tuple((a, p, t - tt) for tt, a, p in recent) if recent else ()

        ctx = HistoricalContext(
            user_txn_count=int(cnt),
            user_amount_mean_idr=mean,
            user_amount_std_idr=std,
            is_new_payer=pid not in seen_payers,
            payer_seen_before=pid in seen_payers,
            payer_age_days=int((t - first_seen) // WINDOW_24H),
            payer_velocity_10m=_count_since(ptimes, t - WINDOW_10M) if ptimes else 0,
            payer_velocity_24h=_count_since(ptimes, t - WINDOW_24H) if ptimes else 0,
            user_velocity_24h=len(recent) if recent else 0,
            country_seen_before=countries[i] in user_countries.get(uid, ()),
            currency_seen_before=currencies[i] in user_currencies.get(uid, ()),
            recent_user_txns=recent_tuples,
        )
        raw = RawTransaction(
            amount_idr=amt,
            payer_id=pid,
            payer_name=payer_names[i],
            hour=int(hours[i]),
            day_of_week=int(days[i]),
        )
        feats = build_features(raw, ctx)
        feats.pop("amount_idr")  # already a raw column on the frame
        feats["hour"] = int(hours[i])
        rows.append(feats)

        # --- update state AFTER emitting (keeps everything past-only) ---
        if ptimes is None:
            payer_times[pid] = [t]
            payer_first_seen[pid] = first_seen
        else:
            ptimes.append(t)
        user_recent.setdefault(uid, deque()).append((t, amt, pid))
        user_countries.setdefault(uid, set()).add(countries[i])
        user_currencies.setdefault(uid, set()).add(currencies[i])
        user_payers.setdefault(uid, set()).add(pid)
        s = user_stats.setdefault(uid, [0.0, 0.0, 0.0])
        s[0] += 1
        delta = amt - s[1]
        s[1] += delta / s[0]
        s[2] += delta * (amt - s[1])

    frame = pd.DataFrame(rows, index=df.index)
    return frame[list(FEATURE_FRAME_COLUMNS)]
