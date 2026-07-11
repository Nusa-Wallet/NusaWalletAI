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

- `synthetic/fraud-synthetic-v1.parquet` — labelled transactions with past-only
  historical/behavioural features (canonical schema in `app/fraud/simulation/schema.py`).
- `synthetic/fraud-synthetic-v1.metadata.json` — seed, git commit, date range, row
  count, anomaly counts per scenario, currency/amount distributions, library versions.

Same seed → identical data. Ten labelled anomaly scenarios (amount spike, velocity
burst, odd hour, new-payer/high-amount, country/currency deviation, duplicate,
invalid identity, account takeover, structuring) at ~5% of rows.
