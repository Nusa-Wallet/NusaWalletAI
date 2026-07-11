# Phase 9 on Local CPU

Phase 9 is zero-shot inference, not training or fine-tuning. Run Chronos-2 and
TimesFM separately so both large models are never held in RAM at the same time.

## Setup

From NusaWalletAI, activate the existing environment and install optional models:

~~~powershell
..\venv\Scripts\Activate.ps1
pip install -r requirements-neural.txt
~~~

Internet is needed only for the first pretrained-model download. Weights are then
read from the local Hugging Face cache.

## Run order

Lightweight pipeline smoke test:

~~~powershell
python scripts\backtest_fx.py --models statistical --max-windows 3 --no-mlflow
~~~

Chronos CPU smoke test:

~~~powershell
python scripts\backtest_fx.py --models chronos-2 --device cpu --batch-size 2 --max-windows 3 --no-mlflow
~~~

TimesFM CPU smoke test, run separately:

~~~powershell
python scripts\backtest_fx.py --models timesfm-2.5 --device cpu --batch-size 2 --max-windows 3 --no-mlflow
~~~

If both succeed, run each complete backtest separately:

~~~powershell
python scripts\backtest_fx.py --models statistical
python scripts\backtest_fx.py --models chronos-2 --device cpu --batch-size 2
python scripts\backtest_fx.py --models timesfm-2.5 --device cpu --batch-size 2
~~~

Use batch-size 1 if memory is tight. CPU runs can take a long time, but no GPU is
required. Close other heavy applications before loading a foundation model.

## Outputs

Each command creates a separate directory below artifacts/fx_backtests containing
predictions.parquet, metrics.json, and metadata.json.

Compare validation metrics when selecting future ensemble weights. Test metrics are
final reporting only and must not drive model selection. Do not commit downloaded
weights, generated predictions, or local MLflow databases.
