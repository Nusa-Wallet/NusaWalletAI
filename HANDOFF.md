# AI Engineering Handoff

Last updated: 2026-07-11.

Read CONTRACTS.md before changing an API and the workspace root README for the
complete product/proposal context.

## Current status

- Phase 0 (API contracts): completed.
- Phase 1 (modular scaffolding): completed and tested.
- Phase 2 (local dependencies and MLflow): committed (requirements-ml.txt). The ML
  deps live in the workspace-root venv (../venv), not the base requirements.txt env.
- Phase 3 (fraud dataset generator): completed and tested. See "Phase 3 result".
- Phase 4 (fraud feature engineering): completed and tested. See "Phase 4 result".
- Phases 5 onward: not implemented.

The working tree contains uncommitted Phase 3-4 changes (app/fraud/simulation,
app/fraud/feature_spec.py, the expanded app/fraud/features.py, the
scripts/generate_fraud_data.py CLI, and the tests/test_fraud_*.py suites). Do not
discard them. Existing backend-compatible endpoints remain POST /fraud/score,
GET /fx/advisory, and GET /health (still the demo model until Phase 7).

The served models are still the demo Isolation Forest/rules and statistical FX
logic. Advanced response fields remain null; do not fabricate forecast values.

## Agreed target models

### Fraud

Target: calibrated CatBoost classifier + Isolation Forest + transparent rules,
combined into an ensemble risk probability and deterministic explanation with
SHAP. CatBoost is primary. LightGBM is only a possible challenger after evaluation.
Use chronological splits, class weighting, calibration, and threshold tuning.

### FX

Candidates: Chronos-2, TimesFM 2.5, and globally trained NHITS, combined only after
walk-forward evaluation into a quantile ensemble and fee-aware decision engine.
Start with zero-shot backtests before foundation-model fine-tuning. Retain a simple
comparator for scientific evaluation even though it is not the intended final model.

## Data strategy

Fraud primary data is a reproducible NusaWallet-specific simulator:

- initial run: 50,000 transactions, 500 users, 2,000 payers, 12 months;
- full run: 200,000-500,000 transactions, 2,000-5,000 users, 18-24 months;
- anomaly ratio approximately 3-7%;
- stable seed and generation metadata are mandatory.

Required scenarios: amount spike, velocity burst, odd hour, new payer/high amount,
country or currency deviation, duplicate payment, invalid identity, account
takeover, and structuring. PaySim is an external benchmark, not rows to concatenate
blindly. AMLSim is a future graph/AML reference, not an MVP dependency.

FX sources are ECB reference rates with Frankfurter as convenient retrieval/fallback.
Target pairs are SGD/IDR, USD/IDR, EUR/IDR, and MYR/IDR. Advanced global training
should use roughly 30-100 pairs and 8-15 years of daily data. Store provenance.
ECB and Frankfurter observations are fallback/cross-check sources, not duplicate
training samples. Use chronological splits and walk-forward evaluation only.

## Compute plan

The developer laptop has no GPU and limited performance.

| Phase | Environment |
| --- | --- |
| 3: small fraud generator | Laptop |
| 3: full fraud generation | Kaggle CPU |
| 4: fraud feature engineering | Hybrid |
| 5: CatBoost training/tuning/calibration | Kaggle GPU |
| 6: SHAP/explainability | Hybrid; full run on Kaggle |
| 7: Fraud FastAPI integration | Laptop |
| 8: FX fetch/preprocessing | Hybrid |
| 9: Chronos/TimesFM backtesting | Kaggle GPU |
| 10: NHITS training | Kaggle GPU |
| 11: foundation-model fine-tuning | Kaggle GPU, only if justified |
| 12: ensemble/decision engine | Hybrid |
| 13-14: backend/mobile integration | Laptop |
| 15: final evaluation/documentation | Hybrid |

Local work covers coding, small samples, schemas, tests, inference, and integration.
Kaggle handles large generation, tuning, backtesting, neural training, and full SHAP.

## Kaggle artifact contract

Expected outputs include fraud_catboost.cbm, fraud_isolation.joblib,
fraud_calibrator.joblib, fraud_metadata.json, fx_nhits.ckpt, and
fx_ensemble_metadata.json.

Every model export must include model/dataset version, feature names and order,
category mappings, preprocessing version, thresholds or ensemble weights, metrics,
Python/dependency versions, dates, and source Git commit.

Generated artifacts and datasets are ignored by Git. Commit schemas, metadata
templates, generator/training code, and tests. Keep Kaggle datasets private unless
explicitly approved; never upload personal/financial data or credentials.

## Phase 3 result (fraud dataset generator)

Implemented under app/fraud/simulation (importable business logic) with a thin CLI:

    ../venv/Scripts/python.exe scripts/generate_fraud_data.py            # full 50k
    ../venv/Scripts/python.exe scripts/generate_fraud_data.py --sample   # fast smoke

Modules: config (constants + SimulationConfig, fixed calendar window and seed),
profiles (user/payer), transactions (normal + shared record/time/amount helpers),
anomalies (the 10 named scenario injectors, round-robin so every type appears),
features (single past-only forward pass; velocities via bisect on sorted per-key
timestamps; documented missing-value policy), schema (CANONICAL_COLUMNS + Pandera),
generator (orchestration + provenance metadata). Outputs land in the gitignored
data/synthetic/ as fraud-synthetic-v1.parquet + .metadata.json.

Definition of done — all met and verified on the root venv:

- same seed produces identical data (assert_frame_equal test);
- IDs unique, amounts positive, currencies supported, timestamps in-window (Pandera);
- every anomaly type represented (round-robin dispatch + test);
- no future or label leakage (features computed before state update; truncation-
  invariance and windowing tests);
- distributions and anomaly counts recorded in metadata JSON;
- 50,000 rows generate and validate end-to-end (5.01% anomalies).

Run tests: ../venv/Scripts/python.exe -m unittest discover -s tests -v (18 pass;
the simulation tests self-skip if pandas/pandera are absent).

Not yet done: full 200k-500k generation on Kaggle CPU (run the same CLI with larger
--rows/--users/--months there).

## Phase 4 result (fraud feature engineering)

Canonical feature set with one shared training/inference code path:

- app/fraud/feature_spec.py — single source of truth: MODEL_FEATURES (16, ordered),
  FEATURE_GROUPS (transaction / behavioural / velocity / identity / geographic),
  MISSING_DEFAULTS (first-transaction policy).
- app/fraud/features.py — stateless transforms (hour_sin/cos, payer_name_quality,
  duplicate_similarity) plus build_features(RawTransaction, HistoricalContext) which
  assembles the exact MODEL_FEATURES vector. Legacy demo helpers retained.
- app/fraud/simulation/features.py — the batch pass now streams per-user/payer state
  and calls build_features per row, so training vectors == what online inference
  (Phase 7) will build. Still strictly past-only.

Schema bumped to fraud-txn-schema-1.1.0; dataset now has 27 columns. New feature
columns added to CANONICAL_COLUMNS and Pandera: hour_sin, hour_cos,
duplicate_similarity, payer_name_quality.

Definition of done — met and verified:

- training and inference use identical code (build_features); a parity test asserts
  the batch row equals a hand-built context result;
- no future/label leakage preserved (state updated after emit; existing tests);
- missing-value policy is explicit (MISSING_DEFAULTS + HistoricalContext defaults);
- per-feature unit tests added; full suite is 27 tests, all passing.
- Regenerated 50k dataset validates; features are discriminative by anomaly type
  (e.g. INVALID_IDENTITY name-quality ~0.03 vs ~1.0 normal; AMOUNT_SPIKE ratio ~7.5),
  zero NaNs.

## Next work: Phase 5

Train the CatBoost fraud model (Kaggle GPU per compute plan): chronological split
(months 1-14 train / 15-18 val / 19-24 test), class weighting, Optuna tuning,
Isolation Forest + rules ensemble, probability calibration, threshold on validation,
single final test evaluation. Consume the MODEL_FEATURES vector; log to MLflow.
Metrics: precision/recall/F1, PR-AUC, ROC-AUC, FPR, Brier, recall per anomaly type.
Note: the 50k local dataset is for wiring; run full 200k-500k generation on Kaggle
first (same CLI, larger --rows/--users/--months).

## Verification and guardrails

Phase 1 ended with Python compilation, four contract tests, both endpoint smoke
requests, OpenAPI generation, and git diff checks passing.

Re-run: python -m unittest discover -s tests -v

- Preserve API compatibility unless introducing a versioned endpoint.
- Share feature logic between Kaggle training and local inference.
- Never train models during FastAPI startup.
- Never present the prototype as processing real money.
- Do not claim profit or superiority without untouched-test/backtest evidence.
- Fraud scores support review; the AI service does not independently settle funds.
