# Phase 10 on Local CPU — Global NHITS

Phase 10 trains one global NHITS across all currency pairs (pair id = series id) on the
TRAIN split only, then evaluates it through the Phase 9 backtest for a fair comparison.
No test data is used for training or selection.

## Setup

~~~powershell
..\venv\Scripts\Activate.ps1
pip install -r requirements-neural.txt   # adds neuralforecast (pulls torch + lightning)
~~~

## Train (CPU)

~~~powershell
python scripts\train_nhits.py --input-size 180 --max-steps 300
~~~

- Trains on the TRAIN split only; the tail (`--val-size`, default 252) is held out for
  early stopping, so Phase 8 val/test are never seen.
- Checkpoint + `train_metadata.json` are written to `artifacts\fx_nhits\<run-id>` and
  logged to MLflow experiment `fx-nhits`.
- ~300 CPU steps is modest but adequate to validate the pipeline. Raise `--max-steps`
  for a stronger model.

Context / seed sweep (the Phase 10 experiment grid) uses the same CLI:

~~~powershell
python scripts\train_nhits.py --input-size 90  --run-id nhits_ctx90
python scripts\train_nhits.py --input-size 365 --run-id nhits_ctx365
python scripts\train_nhits.py --input-size 180 --seed 7 --run-id nhits_ctx180_s7
~~~

## Evaluate (same walk-forward windows as Phase 9)

~~~powershell
python scripts\backtest_fx.py --models nhits --nhits-checkpoint artifacts\fx_nhits\nhits_ctx180
~~~

Select on **validation** metrics only; test metrics are reporting-only. Compare against
the Phase 9 runs (chronos-2, timesfm-2.5, statistical-drift) on the same windows.

## Notes

- torch >= 2.6 defaults `torch.load` to `weights_only=True`; the loader temporarily sets
  `weights_only=False` because the checkpoint is locally produced and trusted.
- Do not commit checkpoints, predictions, or local MLflow databases (all gitignored).
