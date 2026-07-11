# Data workspace

Generated and downloaded datasets are intentionally excluded from Git. Only
reproducible scripts, schemas, and small documentation/sample files should be
committed.

- `raw/`: immutable provider responses and external source files.
- `processed/`: validated, feature-ready Parquet datasets.
- `synthetic/`: reproducible NusaWallet transaction simulations.

Every generated dataset must include metadata recording its version, random seed,
date range, row count, schema version, source/provenance, and generator Git commit.
No real personal or financial data may be committed.
