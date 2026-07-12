"""Phase 11 tests — no chronos, torch download, or training required.

Window building is pure numpy/pandas; the adapter is exercised with a fake pipeline
that mimics Chronos-Bolt's predict_quantiles output.
"""

import unittest

import numpy as np
import pandas as pd

from app.fx.backtest.adapters import create_model
from app.fx.finetune.adapter import ChronosBoltAdapter
from app.fx.finetune.config import TARGET_LENGTH, FinetuneConfig
from app.fx.finetune.data import build_windows


def _panel(n=400):
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    cut1, cut2 = int(n * 0.7), int(n * 0.85)
    split = ["train"] * cut1 + ["val"] * (cut2 - cut1) + ["test"] * (n - cut2)
    frames = []
    for pair, base in [("SGD/IDR", 100.0), ("USD/IDR", 16000.0)]:
        frames.append(pd.DataFrame({"pair": pair, "date": dates,
                                    "rate": base + np.arange(n) * 0.5, "split": split}))
    return pd.concat(frames, ignore_index=True)


class WindowTest(unittest.TestCase):
    def test_windows_shapes_and_train_only(self):
        cfg = FinetuneConfig(context_length=64, n_windows=50, val_windows=20)
        ws = build_windows(_panel(), cfg)
        self.assertEqual(ws.x_train.shape[1], 64)
        self.assertEqual(ws.y_train.shape[1], TARGET_LENGTH)
        self.assertGreater(len(ws.x_train), 0)
        self.assertGreater(len(ws.x_val), 0)
        # A window's target continues its context (contiguity / no scrambling).
        row = ws.x_train[0]
        self.assertTrue(np.all(np.diff(row) > 0))  # increasing synthetic series

    def test_context_too_large_raises(self):
        cfg = FinetuneConfig(context_length=10_000, n_windows=10, val_windows=5)
        with self.assertRaises(ValueError):
            build_windows(_panel(), cfg)


class _FakePipe:
    """Mimics predict_quantiles -> (quantiles[B,H,3], mean[B,H])."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch

        b = len(inputs)
        base = np.array([float(x[-1]) for x in inputs])[:, None]
        h = np.arange(prediction_length)[None, :]
        med = base + h
        q = np.stack([med - 1, med, med + 1], axis=-1)  # (B,H,3)
        return torch.tensor(q), torch.tensor(med)


class AdapterTest(unittest.TestCase):
    def _adapter(self):
        a = ChronosBoltAdapter.__new__(ChronosBoltAdapter)
        a.pipe = _FakePipe()
        a.name, a.version = "chronos-bolt", "test"
        return a

    def test_forecast_batch_maps_quantiles(self):
        a = self._adapter()
        out = a.forecast_batch([np.array([5.0, 6.0, 7.0]), np.array([10.0, 11.0, 12.0])], horizon=3)
        self.assertEqual(out.point.shape, (2, 3))
        self.assertTrue((out.quantiles[0.1] <= out.quantiles[0.5]).all())
        self.assertTrue((out.quantiles[0.5] <= out.quantiles[0.9]).all())
        # second series starts at 12 -> median step 0 = 12
        self.assertEqual(out.quantiles[0.5][1, 0], 12.0)


class RegistrationTest(unittest.TestCase):
    def test_finetuned_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            create_model("chronos-bolt-ft")


if __name__ == "__main__":
    unittest.main()
