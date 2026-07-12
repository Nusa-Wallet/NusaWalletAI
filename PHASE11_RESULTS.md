# Phase 11 Results — Foundation-Model Fine-Tuning

Durable record of the CPU fine-tuning experiment. Checkpoints/predictions under
`artifacts/` are gitignored, so this file is the committed source of truth.

## Setup

- **Model:** Chronos-Bolt-small (47.7M params). The 200M Chronos-2 / TimesFM from Phase 9
  are impractical to fine-tune on CPU; Chronos-Bolt is the CPU-feasible foundation model
  (~1s per train step). This is a smaller sibling of the Phase 9 Chronos-2 — keep that in
  mind across phases.
- **Recipe:** light fine-tune on the TRAIN split only (4,000 windows, context 512,
  target 64), small LR 1e-5, 200 steps, early stopping on a temporally held-out train
  tail (540 windows). ~3.7 min on CPU. Best val loss 24.6 → 22.3.
- **Evaluation:** zero-shot vs fine-tuned Chronos-Bolt through the identical Phase 9/10
  backtest (224 cases each; selection on validation only).

## Validation: zero-shot vs fine-tuned (avg over horizons 1/3/7)

| Pair | zero-shot pinball | fine-tuned pinball | Δ | dir (zs→ft) | pinball winner |
|---|---|---|---|---|---|
| SGD/IDR | 13.11 | 12.84 | −0.27 | 0.476 → 0.417 | fine-tuned |
| USD/IDR | 20.75 | 21.21 | +0.46 | 0.536 → 0.560 | zero-shot |
| EUR/IDR | 33.34 | 31.64 | −1.70 | 0.595 → 0.548 | fine-tuned |
| MYR/IDR | 4.52 | 4.36 | −0.16 | 0.464 → 0.464 | fine-tuned |
| **mean** | **17.93** | **17.51** | **−0.42 (≈2.3%)** | ~0.5 both | 3/4 fine-tuned |

## Verdict — do NOT adopt fine-tuning (per the plan's stopping rule)

Fine-tuning gave a **small, inconsistent** accuracy gain: it improved pinball on 3/4
pairs (~2.3% mean) but **regressed USD/IDR**, and directional accuracy stayed at
coin-flip (~0.5) and even *dropped* on SGD/IDR. The plan's Phase 11 rule is explicit:
*"stop if the improvement over zero-shot is not consistent"* — it isn't. Bigger/adapted
models are not automatically better for FX.

More importantly, this repeats the Phase 9/10 lesson: even a *fine-tuned* model's marginal
accuracy gain does nothing for the real bottleneck — **fee-aware decision quality**, where
the simple statistical-drift comparator still wins. Accuracy is not where the value is.

**Recommendation:** keep zero-shot models (chronos-2 remains the accuracy reference) and
move to **Phase 12 (ensemble + fee-aware decision engine)**, which is what actually turns
forecasts into good convert/hold/split decisions. Revisit fine-tuning only with a GPU and
a decision-aware objective.

## Caveats

- One config (bolt-small, 200 steps, seed 42, LR 1e-5). A longer/GPU run or
  parameter-efficient variant (`--freeze-encoder`) might shift the margins, but the
  decision-quality conclusion is unlikely to change.
- FX output remains a scenario estimate, never guaranteed profit.
