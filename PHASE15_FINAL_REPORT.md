# Phase 15 — Final Evaluation and Documentation

Status: **complete**, with the limitations below. This report consolidates the durable
results from Phases 0–14. Fraud results use a synthetic test set; FX results use historical
ECB reference rates. Neither result proves production performance or future profit.

## Architecture

```text
Mobile (Expo)
  -> Backend (FastAPI: payment, insights, settlement)
       -> AI service (FastAPI)
            -> Fraud: features -> CatBoost + Isolation Forest + rules -> calibration -> risk/explanation
            -> FX: ECB history -> forecasts -> weighted ensemble -> fee-aware decision
       -> safe backend fallback when AI is unavailable
```

The AI service loads frozen artifacts at startup and never trains during requests.
High-risk fraud recommendations are held for review by the backend; the model does not
independently move money. API contracts are documented in `CONTRACTS.md`.

## Dataset cards

### Fraud synthetic v1

- Purpose: controlled development of an imbalanced fraud detector before real labelled
  NusaWallet transactions exist.
- Generator: deterministic seed 42; 50,000 transactions, 500 users, 2,000 payers,
  2025-07-01 through 2026-06-30.
- Labels: 47,497 normal and 2,503 anomalous (5.01%), across ten anomaly scenarios.
- Currency coverage: EUR, MYR, USD, SGD. Amount range after IDR conversion:
  Rp281,772–Rp838,982,550.
- Split: chronological 60/20/20 = 30,000 train, 10,000 validation, 10,000 untouched test.
- Provenance: `data/synthetic/fraud-synthetic-v1.metadata.json`.
- Limitation: simulated patterns can overstate generalisation and do not capture real
  identity, device, merchant, chargeback, or demographic distributions.

### FX dataset v1

- Purpose: global training and rolling-origin evaluation of FX forecasts.
- Source: ECB reference rates via Frankfurter; rates are fetched with EUR base and
  cross-rates are computed locally.
- Coverage: 357,570 rows, 90 ordered pairs from 10 currencies, 2010-12-31–2026-07-10;
  primary pairs SGD/IDR, USD/IDR, EUR/IDR, MYR/IDR.
- Split: chronological 250,290 train, 53,640 validation, 53,640 test; 93 walk-forward windows.
- Integrity: no interpolation over 78 documented TARGET/ECB holiday business dates;
  direct cross-rate verification maximum relative error 0.0036%.
- Provenance: `data/processed/fx-dataset-v1.metadata.json`.
- Limitation: daily reference rates omit intraday volatility, spreads, liquidity, taxes,
  provider outages, and user-specific execution prices.

## Model cards

### Fraud ensemble 1.0.0

- Intended use: merchant-facing risk triage and explanation; not autonomous rejection,
  legal accusation, AML determination, or settlement.
- Components/weights: tuned CatBoost 0.60, Isolation Forest 0.25, transparent rules 0.15;
  calibrated on validation data; decision threshold 0.50.
- Training: seed 42; CatBoost 500 iterations, depth 7, learning rate 0.1222, balanced
  classes; 25-trial random search selected on validation PR-AUC.
- Test metrics: precision 0.9604, recall 0.8466, F1 0.8999, PR-AUC 0.9114,
  ROC-AUC 0.9517, FPR 0.0019, Brier 0.0082.
- Safety: `HIGH` becomes `REVIEW_REQUIRED`; explanations are merchant-facing and must
  not expose exploitable rule thresholds to a payer.

### FX decision engine 1.0.0

- Intended use: historical scenario estimate for convert-now/hold/split assistance.
- Inputs: Chronos-2, TimesFM 2.5, global NHITS, statistical drift; inverse validation
  pinball weights per pair. Model disagreement and interval width reduce confidence.
- Selected system: conservative ensemble decision layer, despite drift having larger
  historical gains, because the latter is a high-variance trend bet.
- Test backtest: 336 cases; total net gain vs immediate conversion +115,932.52 at base
  amount 1,000; mean +345.04; mean confidence 0.728; 90.8% CONVERT_NOW and 9.2% SPLIT.
- Safety: output is an estimate based on history, not a promise that rates will rise or
  that the user will profit.

## Feature definitions

Fraud uses 16 numeric features: IDR-normalised amount; cyclical hour; weekday; payer/user
amount ratio and z-score; duplicate similarity; payer velocity at 10 minutes and 24 hours;
user velocity at 24 hours; new/seen payer flags; payer age/name quality; and whether origin
country/currency were seen previously. Offline training and online inference share the
same feature builder. Missing online history uses documented neutral/default context.

FX uses rate/log return, return lags 1/2/3/5, trailing return mean/std over 5 and 20
observations, weekday, and gap days. Features are trailing-only at date t to prevent
look-ahead leakage.

## Experiment table and ablation

### Fraud (untouched synthetic test)

| Experiment | Precision | Recall | F1 | PR-AUC | FPR | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Rules only | 0.7500 | 0.7806 | 0.7650 | 0.7690 | 0.0141 | 0.0274 |
| Isolation Forest | 0.8491 | 0.6447 | 0.7329 | 0.7921 | 0.0062 | 0.0883 |
| CatBoost baseline | 0.9794 | 0.8291 | 0.8980 | **0.9218** | **0.0009** | 0.0117 |
| CatBoost tuned | 0.9442 | **0.8544** | 0.8970 | 0.9187 | 0.0027 | 0.0124 |
| Full calibrated ensemble | 0.9604 | 0.8466 | **0.8999** | 0.9114 | 0.0019 | **0.0082** |

The ablation shows that the ensemble only narrowly improves F1 over CatBoost, while its
main benefit is calibration (best Brier score). Rules remain valuable as transparent
guardrails. Country and currency deviation recall remain weak (0.2727 and 0.3889), so
production rollout requires real curated signals rather than simulator-specific tuning.

### FX validation comparison (mean pinball; lower is better)

| Model | SGD/IDR | USD/IDR | EUR/IDR | MYR/IDR | Verdict |
|---|---:|---:|---:|---:|---|
| Chronos-2 | **11.81** | **20.71** | **30.49** | 4.45 | accuracy reference |
| TimesFM 2.5 | 12.93 | 21.38 | 31.29 | 4.57 | wide intervals |
| NHITS global | 11.97 | 21.32 | 31.38 | **4.36** | competitive CPU model |
| Statistical drift | 12.82 | 20.77 | 33.30 | 4.42 | high-variance decision comparator |

Chronos-Bolt fine-tuning improved mean pinball only 2.3%, regressed USD/IDR, and left
direction accuracy near 0.5; under the predefined stopping rule it was not adopted.
Forecast-accuracy winners also did not consistently produce better fee-aware decisions.

## Confusion matrix, calibration, and SHAP

The exact confusion matrix and calibration bins are reproducibly generated by:

| Actual / predicted | Normal | Fraud/review |
|---|---:|---:|
| Normal | TN 9,467 | FP 18 |
| Fraud | FN 79 | TP 436 |

```powershell
..\venv\Scripts\python.exe scripts\generate_phase15_report.py
```

Outputs: `reports/phase15/fraud_evaluation.json` and
`reports/phase15/fraud_calibration.svg`. The stored Brier score is 0.0082. Top global
mean absolute SHAP drivers are payer age (1.0919), payer velocity 10m (1.0003), amount
IDR (0.6595), user-relative amount ratio (0.5896), and duplicate similarity (0.2734).
SHAP explains model contribution, not causation or proof of fraud.

## FX walk-forward backtest

Phase 9 evaluated 224 cases/model and 672 horizon predictions/model across four primary
pairs and horizons 1/3/7, selecting only on validation. Detailed tables are retained in
`PHASE9_RESULTS.md` through `PHASE12_RESULTS.md`. The final conservative engine was
positive in aggregate on validation (+38,308.34) and untouched test (+115,932.52), but
MYR/IDR test contribution was negative and directional accuracy remained close to chance.

## Minimum demo and verification

`tests/test_phase15_demo.py` deterministically verifies:

1. normal transaction -> LOW;
2. large new payer -> HIGH / REVIEW_REQUIRED;
3. velocity burst -> REVIEW_REQUIRED;
4. stable FX -> CONVERT_NOW (allowed by HOLD or CONVERT requirement);
5. meaningful forecast with model disagreement -> SPLIT_CONVERSION.

The sixth scenario, AI unavailable -> safe backend fallback, is verified in
`NusaWalletBackend/tests/test_ai_contract.py`: fraud fallback preserves legacy processing
and FX fallback returns a neutral response. These are availability fallbacks, not claims
that an unscored transaction is safe.

## Known limitations

- Fraud is trained and evaluated only on synthetic data; thresholds are not production-ready.
- Online requests provide less history than offline data, so several features default.
- Fraud subgroup fairness cannot be credibly measured without representative real data.
- FX daily reference data and historical backtests do not represent live execution.
- Foundation models were evaluated with limited CPU-scale experiments; NHITS used one seed/config.
- Aggregate FX gain hides per-pair losses and regime dependence; no guaranteed profit.
- Mobile screens were type-checked but not visually/device-tested in the implementation environment.
- Backend fallback prioritises availability and must be monitored; production policy may instead fail closed.

## Ethical and compliance notes

Collect the minimum data needed, document consent/purpose, encrypt sensitive fields, apply
retention and access controls, and never expose payer secrets in explanations. Provide human
review and appeal for held payments; log model/artifact versions and decisions; monitor drift,
false positives, subgroup outcomes, and fallback frequency. Before real deployment, obtain
Indonesian legal/compliance review covering privacy, payment-system, AML/KYC, consumer
protection, and cross-border data obligations. This prototype is decision support, not legal,
financial, or compliance advice and does not process real money.
