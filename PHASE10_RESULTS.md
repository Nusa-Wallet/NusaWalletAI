# Phase 10 Results — Global NHITS

Durable record of the trained global NHITS and its fair comparison against the Phase 9
zero-shot models. Run artifacts under `artifacts/fx_nhits/` and `artifacts/fx_backtests/`
are gitignored, so this file is the committed source of truth.

## Model

- **Type:** one global NHITS (NeuralForecast) across all 90 pairs, pair id = series id.
- **Trained on:** TRAIN split only (250,290 rows); tail (val_size=252) held out for early
  stopping. Phase 8 val/test never seen during training or selection.
- **Config:** input_size 180, horizon 7, MQLoss(level=[80]) → quantiles 0.1/0.5/0.9,
  robust scaling, max_steps 300, seed 42, CPU. (Checkpoint `nhits_ctx180`.)
- **Served via:** `NhitsAdapter` (implements the Phase 9 `ForecastModel` protocol), so it
  runs through the identical rolling-origin backtest and metrics.

## Validation comparison (avg over horizons 1/3/7; selection basis)

`pinball` lower is better; `dir` = directional accuracy; `net_gain` = summed net gain
vs immediate conversion after the 0.5% fee.

| Pair | Model | pinball ↓ | dir | net_gain ↑ |
|---|---|---|---|---|
| SGD/IDR | nhits-global | 11.97 | 0.500 | 0.0 |
| SGD/IDR | **chronos-2** | **11.81** | 0.595 | 1,699.7 |
| SGD/IDR | statistical-drift | 12.82 | 0.524 | **3,002.3** |
| USD/IDR | nhits-global | 21.32 | 0.476 | 0.0 |
| USD/IDR | **chronos-2** | **20.71** | 0.571 | 0.0 |
| USD/IDR | statistical-drift | 20.77 | 0.500 | **18,838.5** |
| EUR/IDR | nhits-global | 31.38 | 0.500 | 0.0 |
| EUR/IDR | **chronos-2** | **30.49** | 0.548 | 0.0 |
| EUR/IDR | statistical-drift | 33.30 | 0.512 | **4,844.6** |
| MYR/IDR | **nhits-global** | **4.36** | 0.429 | 0.0 |
| MYR/IDR | chronos-2 | 4.45 | 0.429 | 0.0 |
| MYR/IDR | statistical-drift | 4.42 | 0.571 | **4,038.2** |

(timesfm-2.5 omitted from the table for brevity; see PHASE9_RESULTS.md — it never cleared
the WAIT threshold.)

## Findings

- **NHITS is competitive on forecast accuracy** despite being a small 300-step CPU model:
  best pinball on MYR/IDR, and within ~2% of chronos-2 on the other pairs.
- **Pinball winners overall:** chronos-2 on 3/4 pairs, nhits-global on MYR/IDR.
- **Directional accuracy ~0.5** for NHITS too — daily FX direction stays near coin-flip.
- **Decision value:** like the foundation models, NHITS rarely produced an expected gain
  above the 0.5% fee, so it defaulted to CONVERT_NOW (net gain 0). **statistical-drift
  still wins fee-aware net gain on all 4 pairs.**

This reinforces the Phase 9 lesson from a second angle: a *trained* global model also
shows that better probabilistic accuracy does not translate into better convert-vs-wait
decisions. **Phase 12 must weight the ensemble/decision engine on fee-aware decision
metrics, not pinball/MAE.**

## Caveats / not done

- One config trained (input_size 180, seed 42, 300 steps). The full experiment grid
  (context 90/180/365 × several seeds) is supported by the CLI but not exhaustively run
  on CPU — a stronger model likely needs more steps / GPU (Phase 11 territory).
- Zero fine-tuning of foundation models yet (Phase 11, only if justified).
- FX output remains a scenario estimate, never guaranteed profit.
