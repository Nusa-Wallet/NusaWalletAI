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
- Phase 5 (CatBoost fraud training): completed and tested. See "Phase 5 result".
- Phase 6 (fraud explainability): completed and tested. See "Phase 6 result".
- Phase 7 (fraud FastAPI integration): completed and tested. See "Phase 7 result".
- Phase 8 (global FX dataset): completed and tested. See "Phase 8 result".
- Phase 9 onward (FX modelling): not implemented.

The fraud track (Phases 3-7) is complete end-to-end. The FX track has started with the
real ECB/Frankfurter dataset (Phase 8). The local MLflow database is intentionally
ignored by Git; portable metrics and provenance belong in model metadata.

FastAPI loads the trained fraud ensemble when compatible artifacts and dependencies
are available. It falls back to the demo Isolation Forest/rules model only when that
bundle cannot be loaded. FX still uses the statistical demo logic; advanced FX
response fields remain null and must not be fabricated before validated models exist.

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

## Phase 5 result (CatBoost fraud training)

Pipeline under app/fraud/training with a thin CLI:

    ../venv/Scripts/python.exe scripts/train_fraud.py            # full run + MLflow
    ../venv/Scripts/python.exe scripts/train_fraud.py --n-trials 40 --no-mlflow

Modules: data (chronological time_split by quantile + to_xy over MODEL_FEATURES),
models (CatBoost with Balanced class weights, IsolationScorer normalised to [0,1],
vectorised transparent rules_score), tuning (Optuna if installed else deterministic
random search — Optuna is NOT in the venv, so it used random search), ensemble
(weighted blend 0.6/0.25/0.15 + isotonic calibration on val + FPR-capped threshold),
evaluate (precision/recall/F1, PR-AUC, ROC-AUC, FPR, Brier, per-anomaly-type recall),
pipeline (runs the 5 experiments, logs MLflow, writes artifacts + metadata).

MLflow: MLflow 3.x dropped the file store, so logging uses the repo sqlite backend
(sqlite:///mlflow.db); override with MLFLOW_TRACKING_URI. 5 runs logged to experiment
"fraud". Best-effort: training still completes and saves artifacts if MLflow fails.

Artifacts (gitignored, in artifacts/): fraud_catboost.cbm, fraud_isolation.joblib,
fraud_calibrator.joblib (holds the EnsembleModel: weights + isotonic calibrator +
threshold), fraud_metadata.json (model/dataset/schema versions, feature names+order,
ensemble weights, threshold, all experiment metrics, tuning info, git commit, seed,
library versions, split row counts). Reload via app.fraud.training.pipeline.load_bundle;
predict_risk() is the shared scoring path for Phase 7.

Test-set results on the 50k dataset (30k/10k/10k chronological split), full-ensemble:
precision 0.96, recall 0.84, F1 0.898, PR-AUC 0.909, ROC-AUC 0.952, FPR 0.0019,
Brier 0.0084. Definition of done — all met: FPR far below the 0.20 target; ensemble
beats rules-only on F1 (0.898 vs 0.739) and PR-AUC; risk probability calibrated
(low Brier); artifacts reload deterministically; reproducible from seed. Per-type
recall is strong for account-takeover/invalid-identity/new-payer/structuring (~1.0)
and weaker for country/currency-deviation (~0.2-0.4) — expected, since those scenarios
overlap legitimate first-time-payer behaviour; revisit in Phase 6.

These metrics measure performance only on the internally generated NusaWallet
synthetic dataset. They do not establish performance on real financial transactions
and must not be presented as production fraud-detection accuracy. External benchmark
and pilot validation remain future work.

Not yet done: full 200k-500k / 18-24 month generation + training on Kaggle GPU (same
CLIs, larger dataset). The month-based 14/4/6 split maps onto time_split fractions.

## Phase 6 result (fraud explainability)

Canonical rules moved to app/fraud/rules_engine.py (vectorised mask + scalar
Indonesian template per rule), shared by the training ensemble (models.rules_score
re-exports it) and the explainability layer. app/fraud/explain/ contains:

- shap_explain.py — TreeSHAP via CatBoost's native ShapValues (no extra runtime dep):
  global_shap_summary + per-row shap_matrix.
- templates.py — Indonesian templates for model (SHAP)-only factors, keyed by topic.
- service.py — explain_transaction / explain_if_flagged: combine triggered rules
  (preferred, value-specific wording) with positive SHAP contributions, de-duplicate
  by topic, rank, cap at 3-5. Flagged transactions always get >=1 reason. No LLM.

Pipeline now computes a global SHAP summary and writes artifacts/fraud_shap_summary.json
(+ top-10 in fraud_metadata.json). Demo: scripts/explain_fraud.py prints factors for a
flagged example per anomaly type.

Definition of done — met: every high-risk prediction has reasons; factors match feature
values (e.g. "Nominal transaksi sekitar 8.7x lebih besar..."); no LLM chooses reasons;
messages are merchant-facing risk reasons, not payer-facing secrets.

Phase 5 flag addressed: added a transparent high_risk_country rule (general FATF-style
list in rules_engine, not the generator's internal set). COUNTRY_DEVIATION recall rose
0.18 -> 0.27 and ensemble F1 0.898 -> 0.900. CURRENCY_DEVIATION stayed ~0.39: those
payers originate from normal countries with modest amounts, so a country-risk list
cannot separate them — the SHAP summary confirms country/currency "seen_before" flags
are low-importance drivers (top drivers: payer_age_days, payer_velocity_10m, amount_idr,
amount_ratio_user). Pushing these types higher would require either a real curated
country/identity signal on real data or gaming the simulator (rejected as dishonest).

## Phase 7 result (fraud FastAPI integration)

app/fraud/inference.py (FraudScorer) loads the bundle once at startup via load_bundle
(never trains). score() builds the feature frame through the SAME build_features /
HistoricalContext path used offline, computes the calibrated ensemble risk, maps to
LOW/MEDIUM/HIGH (+ ALLOW/REVIEW_IF_NEEDED/REVIEW_REQUIRED), and attaches
explain_if_flagged factors. main.py lazy-imports the scorer in lifespan so the API
still starts without ML deps/artifacts (demo fallback); it never trains the real model
at startup. Endpoints: POST /fraud/score (trained or demo), GET /models/fraud/info,
GET /health (now reports fraud_model_loaded). Service version 0.3.0.

Online-context limitation: the request carries only partial history (velocity hints,
is_new_payer); the rest uses the missing-value policy (HistoricalContext defaults), so
amount_ratio/zscore default and country/currency seen_before default False. The backend
should send richer history when available. amount_idr uses the same RATE_TO_IDR as
training so it is comparable.

Definition of done — met: inference needs no retraining; POST /fraud/score and GET
/health stay backward compatible (component_scores.supervised is now populated instead
of null; new fields only added); a parity test asserts the API risk equals the offline
predict_risk on the same frame. Verified live on uvicorn: minimal {amount,currency}
request -> LOW/ALLOW; high-risk request -> HIGH/flagged with 5 Indonesian factors;
/models/fraud/info returns model version + test metrics. Full suite: 53 tests pass.

CONTRACTS.md updated to reflect the served ensemble and the new info endpoint.

## Phase 8 result (global FX dataset)

app/fx/dataset/ builds a reproducible long-panel dataset from real ECB rates via
Frankfurter (endpoint moved to https://api.frankfurter.dev/v1 — must follow redirects).
Modules: config (FxDatasetConfig: 10-currency universe -> 90 ordered pairs, primary
SGD/IDR·USD/IDR·EUR/IDR·MYR/IDR, splits, walk-forward params), providers (EUR-base fetch
+ deterministic synthetic fallback + direct cross-rate cross-check), crossrates
(rate("B/Q") = eur[Q]/eur[B]), features (log_return, return lags, trailing rolling
mean/std, day_of_week, gap_days — all as-of close of day t, leak-free), splits
(chronological by date quantile + rolling-origin walk-forward windows), schema (Pandera,
unique (pair,date)), build (orchestration + provenance/missing-date/verification metadata).

CLI: scripts/fetch_fx_data.py (--start/--end/--sample/--no-verify). Writes raw EUR-base
parquet + provenance to data/raw/ and the processed panel + metadata to data/processed/
(both gitignored). build_dataset accepts a `fetcher` for offline tests.

Real build verified: 357,570 rows, 90 pairs, 2010-12-31..2026-07-10 (~15.5 yrs), 3,973
obs/pair; splits 250,290 / 53,640 / 53,640; 93 walk-forward windows. DoD — all met:
dataset rebuildable; provider provenance recorded (ecb-via-frankfurter vs
synthetic-fallback); no duplicate (pair,date); cross-rates verified against direct API
(max rel error 3.6e-05; EUR/IDR exact); 78 missing ECB-holiday business days documented,
not interpolated. Offline synthetic tests: 10 pass (63 total in the suite).

Provider note: Frankfurter/ECB publishes base-EUR on TARGET business days; cross-rates
are computed locally from that single source so all 90 pairs stay internally consistent.
IDR history via ECB starts ~2011, so that is the practical start of the range.

## Next work: Phase 9 (FX foundation-model backtest)

Zero-shot backtests before any training/fine-tuning: run Chronos-2 and TimesFM 2.5 over
the walk-forward windows (horizons 1/3/7d) on the primary pairs, saving point + quantile
forecasts. Metrics: MAE, RMSE, directional accuracy, quantile loss, interval coverage,
and — crucially — net conversion outcome after the 0.5% fee and maximum regret (decision
quality, not just price error). Log per-pair to MLflow. Keep a simple statistical
comparator (the current advisory logic) as the scientific baseline. These libraries are
heavy (torch etc., requirements-neural.txt) and belong on Kaggle GPU per the compute plan.

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
