"""Phase 9 tests run without Chronos, TimesFM, Torch, network, or GPU."""

import unittest

import numpy as np
import pandas as pd

from app.fx.backtest.contracts import ForecastBatch
from app.fx.backtest.adapters import Chronos2Adapter, TimesFmAdapter
from app.fx.backtest.metrics import compute_metrics
from app.fx.backtest.runner import BacktestConfig, run_backtest


class LinearForecastModel:
    name = "linear-mock"
    version = "test"

    def forecast_batch(self, contexts, horizon):
        points = []
        for context in contexts:
            values = np.asarray(context, dtype=float)
            slope = values[-1] - values[-2]
            points.append(values[-1] + slope * np.arange(1, horizon + 1))
        point = np.asarray(points)
        return ForecastBatch(
            point=point,
            quantiles={0.1: point - 0.1, 0.5: point.copy(), 0.9: point + 0.1},
        )


class ForecastContractTest(unittest.TestCase):
    def test_rejects_crossing_quantiles(self):
        output = ForecastBatch(
            point=np.array([[1.0]]),
            quantiles={
                0.1: np.array([[2.0]]),
                0.5: np.array([[1.0]]),
                0.9: np.array([[3.0]]),
            },
        )
        with self.assertRaises(ValueError):
            output.validate(1, 1)

    def test_chronos_adapter_preserves_numeric_batch_order(self):
        class FakePipeline:
            def predict_df(self, context, **kwargs):
                rows = []
                horizon = kwargs["prediction_length"]
                for series_id in reversed(sorted(context["id"].unique())):
                    for step in range(horizon):
                        value = float(series_id * 10 + step)
                        rows.append({
                            "id": series_id,
                            "timestamp": pd.Timestamp("2030-01-01") + pd.Timedelta(days=step),
                            "predictions": value,
                            "0.1": value - 1,
                            "0.5": value,
                            "0.9": value + 1,
                        })
                return pd.DataFrame(rows)

        adapter = Chronos2Adapter.__new__(Chronos2Adapter)
        adapter.pipeline = FakePipeline()
        contexts = [np.array([1.0, 2.0])] * 12
        output = adapter.forecast_batch(contexts, 2)
        self.assertEqual(output.point[2, 0], 20.0)
        self.assertEqual(output.point[10, 0], 100.0)

    def test_timesfm_adapter_maps_official_quantile_positions(self):
        class FakeModel:
            def forecast(self, horizon, inputs):
                batch = len(inputs)
                point = np.ones((batch, horizon))
                quantiles = np.zeros((batch, horizon, 10))
                for index in range(10):
                    quantiles[:, :, index] = index
                return point, quantiles

        adapter = TimesFmAdapter.__new__(TimesFmAdapter)
        adapter.model = FakeModel()
        output = adapter.forecast_batch([np.array([1.0, 2.0])], 2)
        self.assertTrue((output.quantiles[0.1] == 1).all())
        self.assertTrue((output.quantiles[0.5] == 5).all())
        self.assertTrue((output.quantiles[0.9] == 9).all())


class MetricsTest(unittest.TestCase):
    def test_perfect_forecast_metrics(self):
        frame = pd.DataFrame({
            "actual_rate": [101.0, 102.0],
            "point_forecast": [101.0, 102.0],
            "current_rate": [100.0, 101.0],
            "q10": [100.0, 101.0],
            "q50": [101.0, 102.0],
            "q90": [102.0, 103.0],
            "action": ["WAIT", "WAIT"],
            "immediate_net": [99.5, 100.495],
            "chosen_net": [100.495, 101.49],
            "net_gain_vs_immediate": [0.995, 0.995],
            "regret": [0.0, 0.0],
        })
        metrics = compute_metrics(frame)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["directional_accuracy"], 1.0)
        self.assertEqual(metrics["interval_coverage_80"], 1.0)


class RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        split = ["train"] * 30 + ["val"] * 15 + ["test"] * 15
        cls.panel = pd.DataFrame({
            "pair": "SGD/IDR",
            "date": dates,
            "rate": 100.0 + np.arange(60),
            "split": split,
        })
        cls.metadata = {
            "dataset_version": "test",
            "schema_version": "test",
            "walk_forward": {
                "windows": [
                    {
                        "train_start": dates[0].date().isoformat(),
                        "train_end": dates[29].date().isoformat(),
                        "test_start": dates[30].date().isoformat(),
                        "test_end": dates[36].date().isoformat(),
                        "horizon_obs": 7,
                    },
                    {
                        "train_start": dates[0].date().isoformat(),
                        "train_end": dates[44].date().isoformat(),
                        "test_start": dates[45].date().isoformat(),
                        "test_end": dates[51].date().isoformat(),
                        "horizon_obs": 7,
                    },
                ]
            },
        }

    def test_walk_forward_keeps_validation_and_test_separate(self):
        predictions, metadata = run_backtest(
            self.panel,
            self.metadata,
            [LinearForecastModel()],
            BacktestConfig(
                pairs=("SGD/IDR",),
                horizons=(1, 3, 7),
                context_length=20,
                batch_size=2,
            ),
        )
        self.assertEqual(len(predictions), 6)
        self.assertEqual(set(predictions["evaluation_split"]), {"val", "test"})
        self.assertEqual(set(predictions["horizon"]), {1, 3, 7})
        self.assertTrue((predictions["point_forecast"] == predictions["actual_rate"]).all())
        self.assertEqual(len(metadata["metrics"]), 6)
        self.assertTrue(all(values["mae"] == 0 for values in metadata["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
