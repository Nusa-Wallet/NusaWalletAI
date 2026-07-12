# Pipeline entry points

This directory holds reproducible data, training, and evaluation commands.
Importable business logic belongs under `app/`; scripts only parse arguments and
invoke that logic.

Run with the workspace-root venv (`../venv/Scripts/python.exe`):

- `generate_fraud_data.py` — synthetic fraud dataset (Phase 3-4). See `../data/README.md`.
- `train_fraud.py` — fraud ensemble training (Phase 5): trains CatBoost + Isolation
  Forest + rules, calibrates, evaluates on a chronological test split, logs to MLflow
  (`sqlite:///mlflow.db`), and writes artifacts to `../artifacts/` (incl. the SHAP
  global summary).
- `explain_fraud.py` — explainability demo (Phase 6): loads the trained bundle and
  prints 3-5 Indonesian risk factors for a flagged example per anomaly type.
- `fetch_fx_data.py` — global FX dataset (Phase 8): fetches real ECB rates via
  Frankfurter, computes cross-rates + features + chronological splits + walk-forward
  windows, and writes `../data/raw/` (EUR-base + provenance) and `../data/processed/`
  (panel + metadata). `--sample`/`--no-verify` for a fast offline run.

- `backtest_fx.py` — Phase 9/10 batched backtest. Zero-shot (statistical, chronos-2,
  timesfm-2.5) and the trained NHITS (`--models nhits --nhits-checkpoint <dir>`) all run
  through the same rolling-origin windows. See `../LOCAL_PHASE9.md`.
- `train_nhits.py` — Phase 10 global NHITS training (CPU): trains on the TRAIN split,
  saves a checkpoint to `../artifacts/fx_nhits/`, logs to MLflow. See `../LOCAL_PHASE10.md`
  and `../requirements-neural.txt`.
