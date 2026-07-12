# Phase 11 on Local CPU — Foundation-Model Fine-Tuning

Phase 11 fine-tunes a foundation model on our FX data and compares it to zero-shot on
the same walk-forward windows. The 200M Chronos-2 / TimesFM (Phase 9) are impractical to
fine-tune on CPU, so we fine-tune **Chronos-Bolt-small** (47.7M) — small enough that a
CPU training step is ~1s.

## Setup

~~~powershell
..\venv\Scripts\Activate.ps1
pip install -r requirements-neural.txt   # chronos-forecasting already includes bolt
~~~

## Fine-tune (CPU)

~~~powershell
python scripts\finetune_fx.py --max-steps 200
~~~

- Trains on the TRAIN split only; the latest windows per pair are held out (with a gap)
  for early stopping, so Phase 8 val/test are never seen.
- Light recipe: small LR (1e-5), capped steps, early stopping. `--freeze-encoder` for a
  parameter-efficient variant (train only the decoder/head).
- Checkpoint (`save_pretrained`) + `train_metadata.json` land in
  `artifacts\fx_finetune\<run-id>`; logged to MLflow experiment `fx-finetune`.
- First run downloads `amazon/chronos-bolt-small` (~cached afterwards).

## Compare zero-shot vs fine-tuned (same windows as Phase 9/10)

~~~powershell
python scripts\backtest_fx.py --models chronos-bolt --no-mlflow
python scripts\backtest_fx.py --models chronos-bolt-ft --checkpoint artifacts\fx_finetune\<run-id> --no-mlflow
~~~

Select on **validation** only. Per the plan, **stop if the improvement over zero-shot is
not consistent** — bigger/adapted models are not automatically better for FX.

## Notes

- This is Chronos-Bolt, a smaller sibling of the Phase 9 Chronos-2, chosen for CPU
  feasibility — keep that in mind when comparing across phases.
- Do not commit checkpoints, downloaded weights, predictions, or local MLflow DBs.
