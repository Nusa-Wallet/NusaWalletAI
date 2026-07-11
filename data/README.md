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

## Synthetic fraud dataset (Phase 3)

Reproducible generator: `app/fraud/simulation` (logic) + `scripts/generate_fraud_data.py`
(CLI). Requires the ML deps in the workspace-root venv (`../venv`).

```powershell
..\venv\Scripts\python.exe scripts\generate_fraud_data.py            # full 50k rows
..\venv\Scripts\python.exe scripts\generate_fraud_data.py --sample   # fast smoke run
```

Outputs (both gitignored):

- `synthetic/fraud-synthetic-v1.parquet` — labelled transactions with the past-only
  canonical feature set (`app/fraud/feature_spec.py`: transaction / behavioural /
  velocity / identity / geographic groups; column order in
  `app/fraud/simulation/schema.py`). Training and online inference build these via the
  same `app.fraud.features.build_features`.
- `synthetic/fraud-synthetic-v1.metadata.json` — seed, git commit, date range, row
  count, anomaly counts per scenario, currency/amount distributions, library versions.

Same seed → identical data. Ten labelled anomaly scenarios (amount spike, velocity
burst, odd hour, new-payer/high-amount, country/currency deviation, duplicate,
invalid identity, account takeover, structuring) at ~5% of rows.

## Global FX dataset (Phase 8)

Reproducible builder: `app/fx/dataset` + `scripts/fetch_fx_data.py`. Real ECB reference
rates via Frankfurter (`https://api.frankfurter.dev/v1`, follow redirects); deterministic
synthetic fallback offline. Requires the root venv (`../venv`).

```powershell
..\venv\Scripts\python.exe scripts\fetch_fx_data.py            # ~15y, 90 pairs, real data
..\venv\Scripts\python.exe scripts\fetch_fx_data.py --sample   # small offline run
```

Outputs (gitignored):

- `raw/fx-dataset-v1_eur_base.parquet` + `raw/fx-dataset-v1_provenance.json` — the raw
  EUR-based rates and provider provenance (real vs synthetic).
- `processed/fx-dataset-v1.parquet` — long panel `[pair, date, rate, log_return, return
  lags, rolling mean/std, day_of_week, gap_days, split]` for the 90 cross pairs.
- `processed/fx-dataset-v1.metadata.json` — provenance, cross-rate verification, missing
  ECB-holiday dates, split counts, and walk-forward windows.

Cross-rate for `"BASE/QUOTE"` = `eur[QUOTE]/eur[BASE]`; features are as-of close of day
t (leak-free). Missing business days (ECB holidays) are documented, not interpolated.
