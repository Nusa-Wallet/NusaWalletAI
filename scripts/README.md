# Pipeline entry points

This directory holds reproducible data, training, and evaluation commands.
Importable business logic belongs under `app/`; scripts only parse arguments and
invoke that logic.

Run with the workspace-root venv (`../venv/Scripts/python.exe`):

- `generate_fraud_data.py` — synthetic fraud dataset (Phase 3-4). See `../data/README.md`.
- `train_fraud.py` — fraud ensemble training (Phase 5): trains CatBoost + Isolation
  Forest + rules, calibrates, evaluates on a chronological test split, logs to MLflow
  (`sqlite:///mlflow.db`), and writes artifacts to `../artifacts/`.

Evaluation/FX-pipeline scripts are scheduled for later phases.
