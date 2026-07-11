"""Configuration for the global FX dataset build."""

from dataclasses import dataclass, replace
from datetime import date

DATASET_VERSION = "fx-dataset-v1"
SCHEMA_VERSION = "fx-panel-schema-1.0.0"

# ECB reference rates are published base-EUR on TARGET business days; Frankfurter
# is the retrieval layer. The .dev host is the current canonical endpoint.
FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v1"
PROVIDER_REAL = "ecb-via-frankfurter"
PROVIDER_FALLBACK = "synthetic-fallback"

# Currency universe (subset of the 30 ECB currencies): majors + SEA corridors
# relevant to Indonesian digital exporters. 10 currencies -> 90 ordered pairs.
DEFAULT_CURRENCIES = ("EUR", "USD", "JPY", "GBP", "SGD", "MYR", "IDR", "THB", "PHP", "INR")
PRIMARY_PAIRS = ("SGD/IDR", "USD/IDR", "EUR/IDR", "MYR/IDR")


@dataclass(frozen=True)
class FxDatasetConfig:
    currencies: tuple[str, ...] = DEFAULT_CURRENCIES
    primary_pairs: tuple[str, ...] = PRIMARY_PAIRS
    base: str = "EUR"
    start_date: str = "2011-01-01"
    end_date: str | None = None  # defaults to today

    train_frac: float = 0.70
    val_frac: float = 0.15

    return_lags: tuple[int, ...] = (1, 2, 3, 5)
    rolling_windows: tuple[int, ...] = (5, 20)

    # Walk-forward windows are expressed in observation (business-day) counts.
    wf_initial_train_obs: int = 8 * 252  # ~8 years
    wf_horizon_obs: int = 7              # forecast horizon (matches Phase 9: 1/3/7d)
    wf_step_obs: int = 21               # roll roughly monthly

    seed: int = 42  # deterministic synthetic fallback only

    dataset_version: str = DATASET_VERSION
    schema_version: str = SCHEMA_VERSION

    def end(self) -> str:
        return self.end_date or date.today().isoformat()

    def symbols(self) -> list[str]:
        """Currencies to request from the base (everything except the base)."""
        return [c for c in self.currencies if c != self.base]

    def pairs(self) -> list[str]:
        """All ordered cross pairs among the universe, primary pairs first."""
        ordered = [f"{b}/{q}" for b in self.currencies for q in self.currencies if b != q]
        primary = [p for p in self.primary_pairs if p in ordered]
        rest = [p for p in ordered if p not in primary]
        return primary + rest

    @classmethod
    def small_sample(cls, **overrides) -> "FxDatasetConfig":
        base = cls(
            currencies=("EUR", "USD", "SGD", "IDR", "MYR"),
            start_date="2020-01-01",
            end_date="2021-12-31",
            wf_initial_train_obs=200,
            wf_horizon_obs=7,
            wf_step_obs=30,
        )
        return replace(base, **overrides) if overrides else base
