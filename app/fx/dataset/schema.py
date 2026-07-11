"""Canonical FX panel schema and Pandera validation."""

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

from app.fx.dataset.config import FxDatasetConfig
from app.fx.dataset.features import feature_columns


def canonical_columns(config: FxDatasetConfig) -> list[str]:
    return ["pair", "date", "rate", *feature_columns(config), "split"]


def build_schema(config: FxDatasetConfig) -> "pa.DataFrameSchema":
    nullable_float = {"nullable": True}
    columns = {
        "pair": pa.Column(str, pa.Check.str_contains("/")),
        "date": pa.Column("datetime64[ns, UTC]"),
        "rate": pa.Column(float, pa.Check.gt(0)),
        "log_return": pa.Column(float, **nullable_float),
        "day_of_week": pa.Column(int, pa.Check.in_range(0, 6)),
        "gap_days": pa.Column(float, pa.Check.ge(0), **nullable_float),
        "split": pa.Column(str, pa.Check.isin(["train", "val", "test"])),
    }
    for lag in config.return_lags:
        columns[f"return_lag_{lag}"] = pa.Column(float, **nullable_float)
    for window in config.rolling_windows:
        columns[f"roll_mean_{window}"] = pa.Column(float, **nullable_float)
        columns[f"roll_std_{window}"] = pa.Column(float, pa.Check.ge(0), **nullable_float)

    return pa.DataFrameSchema(
        columns,
        strict=True,
        ordered=False,
        unique=["pair", "date"],
        coerce=True,
    )


def validate(panel: pd.DataFrame, config: FxDatasetConfig) -> pd.DataFrame:
    return build_schema(config).validate(panel, lazy=True)
