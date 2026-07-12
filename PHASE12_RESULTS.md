# Phase 12 Results — FX Ensemble + Fee-Aware Decision Engine

Durable record of the ensemble weighting and decision-engine evaluation. The
`artifacts/fx_ensemble/fx_ensemble_metadata.json` output is gitignored, so this file is
the committed source of truth. Built by combining the existing Phase 9/10 predictions
(no model re-run) with `scripts/build_ensemble.py`.

## Ensemble

- **Models combined:** chronos-2, timesfm-2.5, nhits-global, statistical-drift.
- **Weights:** per-pair, inverse validation pinball (averaged over horizons 1/3/7),
  normalised per pair. Example SGD/IDR ≈ {chronos-2 0.262, nhits 0.258, statistical
  0.241, timesfm 0.239} — accuracy-weighted, so the forecast distribution is dominated
  by the accurate (near-current) forecasters.
- **Disagreement** = std of model point forecasts → the confidence signal.

## Decision engine

Turns the ensemble forecast (point + interval) + disagreement + amount + risk preference
into the frozen `FxAdvisoryResponse` shape: action (CONVERT_NOW / HOLD_TEMPORARILY /
SPLIT_CONVERSION), confidence, forecast range, `recommended_convert_percentage`,
`estimated_gain_loss` (from the user's amount), scenario best/worst, and Indonesian
rationale + reasons. Confidence falls with model disagreement and interval width.

Fee handling (corrected from the Phase 9 note): the 0.5% fee is paid on any conversion,
so it is **neutral to now-vs-later timing**. The engine therefore holds on a positive
expected move (scaled so a fee-sized 0.5% move is "small"), gated by confidence; the fee
enters the gain estimate and the meaningful-move scale, not as a hard hurdle.

Example (SPLIT): SGD/IDR 12,000 → forecast 12,180 (+1.5%), interval 12,050–12,310,
moderate disagreement → **SPLIT_CONVERSION, 39% now, confidence 0.61, est. gain
Rp109,251 on 1,000**, with an Indonesian rationale.

## Evaluation — fee-aware net gain (amount 1,000, MODERATE)

| Split | total net gain | mean | actions | mean confidence |
|---|---|---|---|---|
| validation | +38,308 | +114 | 96% CONVERT_NOW / 4% SPLIT | 0.73 |
| **test** | **+115,932** | **+345** | 91% CONVERT_NOW / 9% SPLIT | 0.73 |

Per-model net gain, for context (their own backtest decisions):

| Model | val net gain | test net gain |
|---|---|---|
| statistical-drift | +860,261 | +1,794,742 |
| chronos-2 | +47,593 | 0 |
| timesfm-2.5 | −172,244 | −347,105 |
| nhits-global | 0 | 0 |
| **ensemble engine** | **+38,308** | **+115,932** |

## Findings

- **The engine is deliberately conservative and generalises:** positive net gain on both
  validation and test, mostly CONVERT_NOW with selective SPLIT when confident, low regret
  variance.
- **statistical-drift's much larger gains are a high-variance directional bet.** It made
  money by aggressively extrapolating IDR's depreciation trend (which persisted 2011–2026),
  but the same aggression made timesfm-2.5 *lose* heavily (−347k test). That is not a
  responsible default recommendation for users.
- The accuracy-weighted ensemble forecasts are near-current, so the fee-aware engine only
  holds/splits on genuine, confident signals — consistent with the mandate to present FX
  as scenario estimates, never guaranteed profit.

## Definition of done — met

- Split percentage available (SPLIT_CONVERSION with `recommended_convert_percentage`).
- Confidence falls when models disagree (unit-tested and reflected in outputs).
- Gain/loss computed from the user's nominal amount, net of fee.
- Rationale derived from model data (expected move, confidence) and the decision rule.
- Fee accounted for; ensemble weighted per pair on validation error.

## Caveats / next

- Live serving (running the models per request to feed the engine) is Phase 13, which
  also wires this into GET /fx/advisory (currently still the legacy statistical service).
- A more aggressive, trend-following mode is possible but is a higher-risk product/risk
  choice, not a modelling one — keep the conservative default.
