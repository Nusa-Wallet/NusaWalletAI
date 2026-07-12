"""Phase 10 NHITS tests — no neuralforecast, torch, or training required.

The training-frame builder is pure pandas; the adapter is exercised with a fake
NeuralForecast whose predict() mimics the real MQLoss column layout.
"""

import unittest

import numpy as np
import pandas as pd

from app.fx.backtest.adapters import create_model
from app.fx.nhits.adapter import NhitsAdapter
from app.fx.nhits.training import to_training_frame


def _panel():
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    split = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    frames = []
    for pair, base in [("SGD/IDR", 100.0), ("USD/IDR", 16000.0)]:
        frames.append(pd.DataFrame({"pair": pair, "date": dates,
                                    "rate": base + np.arange(30), "split": split}))
    return pd.concat(frames, ignore_index=True)


class TrainingFrameTest(unittest.TestCase):
    def test_uses_train_split_only_with_synthetic_index(self):
        frame = to_training_frame(_panel())
        self.assertEqual(set(frame.columns), {"unique_id", "ds", "y"})
        self.assertEqual(set(frame["unique_id"]), {"SGD/IDR", "USD/IDR"})
        # 20 train rows per pair (val/test excluded)
        self.assertEqual(len(frame), 40)
        sgd = frame[frame["unique_id"] == "SGD/IDR"].reset_index(drop=True)
        self.assertEqual(sgd["y"].iloc[0], 100.0)          # first train rate
        self.assertEqual(sgd["y"].iloc[-1], 119.0)         # last train rate (index 19)
        # contiguous daily synthetic dates
        self.assertEqual((sgd["ds"].diff().dropna() == pd.Timedelta(days=1)).all(), True)

    def test_pairs_filter(self):
        frame = to_training_frame(_panel(), pairs=("USD/IDR",))
        self.assertEqual(set(frame["unique_id"]), {"USD/IDR"})


class _FakeNF:
    """Mimics NeuralForecast.predict with MQLoss(level=[80]) columns, uid as index."""

    def __init__(self, horizon=7, crossing=False):
        self.horizon = horizon
        self.crossing = crossing

    def predict(self, df):
        rows = []
        for uid in df["unique_id"].unique():
            median = 100.0 + int(uid)
            lo, hi = (median + 1, median - 1) if self.crossing else (median - 1, median + 1)
            for step in range(self.horizon):
                rows.append({"unique_id": uid,
                             "ds": pd.Timestamp("2001-01-01") + pd.Timedelta(days=step),
                             "NHITS-median": median, "NHITS-lo-80": lo, "NHITS-hi-80": hi})
        return pd.DataFrame(rows).set_index("unique_id")


class AdapterTest(unittest.TestCase):
    def _adapter(self, **kw):
        adapter = NhitsAdapter.__new__(NhitsAdapter)
        adapter.nf = _FakeNF(**kw)
        return adapter

    def test_parses_median_and_quantiles_in_order(self):
        adapter = self._adapter()
        contexts = [np.linspace(1, 10, 200) for _ in range(3)]
        out = adapter.forecast_batch(contexts, horizon=3)
        self.assertEqual(out.point.shape, (3, 3))
        # uid order preserved (str(index) -> int): row 1 -> median 101
        self.assertTrue((out.point[1] == 101.0).all())
        self.assertTrue((out.quantiles[0.1][1] == 100.0).all())
        self.assertTrue((out.quantiles[0.9][1] == 102.0).all())

    def test_enforces_monotone_quantiles(self):
        adapter = self._adapter(crossing=True)  # lo>hi from the model
        out = adapter.forecast_batch([np.linspace(1, 10, 200)], horizon=2)
        self.assertTrue((out.quantiles[0.1] <= out.quantiles[0.5]).all())
        self.assertTrue((out.quantiles[0.5] <= out.quantiles[0.9]).all())

    def test_detect_columns_rejects_unexpected(self):
        with self.assertRaises(RuntimeError):
            NhitsAdapter._detect_columns(["ds", "foo", "bar"])


class RegistrationTest(unittest.TestCase):
    def test_nhits_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            create_model("nhits")


if __name__ == "__main__":
    unittest.main()
