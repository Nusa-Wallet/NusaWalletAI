# Phase 9 Results — FX Zero-Shot Backtest

Durable record of the executed zero-shot backtests. The run artifacts under
`artifacts/fx_backtests/` are gitignored, so these tables are the committed source of
truth (reproduce with `scripts/backtest_fx.py`).

- **Models:** chronos-2, timesfm-2.5, statistical-drift (comparator).
- **Setup:** 224 rolling-origin cases per model → 672 predictions; primary pairs
  SGD/IDR, USD/IDR, EUR/IDR, MYR/IDR; horizons 1/3/7; fee 0.5%; base amount 1000.
- **Dataset:** fx-dataset-v1 (real ECB via Frankfurter, 2011–2026).
- **Selection basis:** validation split only. Test metrics are reporting-only and were
  not used to pick a model.

## Validation comparison (averaged over horizons 1/3/7)

`pinball` = mean pinball loss (lower better); `dir` = directional accuracy;
`cov` = 80% interval coverage (nominal 0.80); `net_gain` = summed net gain vs
immediate conversion after fee (higher better).

| Pair | Model | pinball ↓ | dir | cov | net_gain ↑ |
|---|---|---|---|---|---|
| SGD/IDR | **chronos-2** | **11.81** | 0.595 | 0.821 | 1,699.7 |
| SGD/IDR | timesfm-2.5 | 12.93 | 0.464 | 0.929 | 0.0 |
| SGD/IDR | statistical-drift | 12.82 | 0.524 | 0.857 | **3,002.3** |
| USD/IDR | **chronos-2** | **20.71** | 0.571 | 0.833 | 0.0 |
| USD/IDR | timesfm-2.5 | 21.38 | 0.655 | 0.905 | 0.0 |
| USD/IDR | statistical-drift | 20.77 | 0.500 | 0.869 | **18,838.5** |
| EUR/IDR | **chronos-2** | **30.49** | 0.548 | 0.762 | 0.0 |
| EUR/IDR | timesfm-2.5 | 31.29 | 0.512 | 0.833 | -6,151.6 |
| EUR/IDR | statistical-drift | 33.30 | 0.512 | 0.738 | **4,844.6** |
| MYR/IDR | chronos-2 | 4.45 | 0.429 | 0.786 | 0.0 |
| MYR/IDR | timesfm-2.5 | 4.57 | 0.512 | 0.833 | 0.0 |
| MYR/IDR | **statistical-drift** | **4.42** | 0.571 | 0.857 | **4,038.2** |

## Winners

- **Forecast accuracy (pinball):** chronos-2 on 3/4 pairs; statistical-drift on MYR/IDR.
- **Fee-aware decision value (net gain):** statistical-drift on all 4 pairs.
- **Directional accuracy:** ~0.5 (coin-flip) for every model — daily FX direction is
  not reliably predictable here.
- **Interval coverage:** timesfm-2.5 has the widest, best-covered intervals (~0.83–0.93)
  but never clears the WAIT threshold, so its decision value is 0.

## Key finding (feeds Phase 10/12)

The most accurate probabilistic forecaster (chronos-2) is **not** the best decision
maker. The foundation models rarely produced an expected gain above the 0.5% fee, so
they defaulted to CONVERT_NOW (net gain 0), while the simple drift comparator captured
the positive net gain. **Model selection for the Phase 12 decision engine must be driven
by fee-aware decision metrics (net gain / regret), not pinball or MAE alone.** A useful
split of roles: chronos-2 for the forecast distribution, a drift/decision layer for the
convert-vs-wait call.

## Caveats

- Zero-shot only; no fine-tuning (that is Phase 11, and only if justified).
- Directional accuracy near 0.5 means these are weak point predictors — treat FX output
  as scenario estimates, never guaranteed profit.
- Decision economics simplification: the 0.5% fee is applied to both immediate and
  waited conversion, so it acts as a WAIT hurdle rather than an asymmetric cost;
  revisit when building the real fee-aware decision engine (Phase 12).
