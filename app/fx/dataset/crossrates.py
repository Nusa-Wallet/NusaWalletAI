"""Cross-rate computation from EUR-based rates.

For a pair ``"BASE/QUOTE"`` the rate is QUOTE units per 1 BASE, derived from the shared
EUR-based frame as ``eur[QUOTE] / eur[BASE]`` (EUR cancels). This keeps every pair
internally consistent; ``providers.verify_cross_rates`` cross-checks the primary pairs
against direct API fetches.
"""

import pandas as pd


def compute_long_panel(eur_frame: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    """Return a long panel [pair, date, rate] for the requested cross pairs."""
    parts: list[pd.DataFrame] = []
    for pair in pairs:
        base, quote = pair.split("/")
        if base not in eur_frame.columns or quote not in eur_frame.columns:
            continue
        rate = (eur_frame[quote] / eur_frame[base]).dropna()
        parts.append(pd.DataFrame({"pair": pair, "date": rate.index, "rate": rate.to_numpy()}))
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["pair", "date"], kind="mergesort").reset_index(drop=True)
