"""Phase 12 tests: fee-aware decision engine, per-pair weights, ensemble combine."""

import unittest

import numpy as np
import pandas as pd

from app.fx.decision.config import DecisionConfig
from app.fx.decision.engine import decide
from app.fx.decision.ensemble import combine_predictions
from app.fx.decision.weights import derive_weights


def _base(**kw):
    args = dict(pair="SGD/IDR", current_rate=10000.0, forecast_rate=10000.0,
                forecast_lower=9950.0, forecast_upper=10050.0, disagreement=0.0,
                amount=1000.0, horizon_days=7, risk_preference="MODERATE")
    args.update(kw)
    return decide(**args)


class EngineTest(unittest.TestCase):
    def test_confident_appreciation_holds(self):
        d = _base(forecast_rate=10200.0, forecast_lower=10150.0, forecast_upper=10250.0)
        self.assertIn(d.action, {"HOLD_TEMPORARILY", "SPLIT_CONVERSION"})
        self.assertLess(d.recommended_convert_percentage, 100)
        self.assertGreater(d.estimated_gain_loss, 0)

    def test_depreciation_converts_now(self):
        d = _base(forecast_rate=9800.0, forecast_lower=9750.0, forecast_upper=9850.0)
        self.assertEqual(d.action, "CONVERT_NOW")
        self.assertEqual(d.recommended_convert_percentage, 100)

    def test_small_gain_holds_only_a_little(self):
        # +0.2% expected (small vs the 0.5% fee scale) -> mostly convert now
        d = _base(forecast_rate=10020.0)
        self.assertGreaterEqual(d.recommended_convert_percentage, 70)
        self.assertIn(d.action, {"CONVERT_NOW", "SPLIT_CONVERSION"})

    def test_confidence_falls_with_disagreement(self):
        low = _base(disagreement=1.0)
        high = _base(disagreement=300.0)
        self.assertLess(high.confidence, low.confidence)

    def test_wide_interval_lowers_confidence(self):
        narrow = _base(forecast_lower=9990.0, forecast_upper=10010.0)
        wide = _base(forecast_lower=9500.0, forecast_upper=10500.0)
        self.assertLess(wide.confidence, narrow.confidence)

    def test_gain_scales_with_amount(self):
        small = _base(forecast_rate=10200.0, amount=1000.0)
        big = _base(forecast_rate=10200.0, amount=10000.0)
        self.assertAlmostEqual(big.estimated_gain_loss, small.estimated_gain_loss * 10, places=2)

    def test_risk_preference_orders_conversion(self):
        kw = dict(forecast_rate=10200.0, forecast_lower=10150.0, forecast_upper=10250.0)
        conservative = _base(risk_preference="CONSERVATIVE", **kw)
        aggressive = _base(risk_preference="AGGRESSIVE", **kw)
        # conservative converts more now (holds less)
        self.assertGreaterEqual(conservative.recommended_convert_percentage,
                                aggressive.recommended_convert_percentage)

    def test_contract_fields_present(self):
        d = _base().to_dict()
        for key in ("pair", "action", "confidence", "current_rate", "forecast_rate",
                    "forecast_lower", "forecast_upper", "recommended_convert_percentage",
                    "estimated_gain_loss", "scenario_best", "scenario_worst",
                    "rationale", "reasons", "model_version"):
            self.assertIn(key, d)
        self.assertIn(d["action"], {"CONVERT_NOW", "HOLD_TEMPORARILY", "SPLIT_CONVERSION"})

    def test_amount_none_gives_null_gain(self):
        self.assertIsNone(_base(amount=None).estimated_gain_loss)


class WeightsTest(unittest.TestCase):
    def test_inverse_error_weighting(self):
        metrics = {
            "a": {f"a|SGD/IDR|val|h{h}": {"mean_pinball": 10.0} for h in (1, 3, 7)},
            "b": {f"b|SGD/IDR|val|h{h}": {"mean_pinball": 20.0} for h in (1, 3, 7)},
        }
        w = derive_weights(metrics, ("a", "b"), ["SGD/IDR"])["SGD/IDR"]
        self.assertAlmostEqual(w["a"] + w["b"], 1.0, places=5)
        self.assertGreater(w["a"], w["b"])  # lower error -> higher weight
        self.assertAlmostEqual(w["a"], 2 / 3, places=3)


class EnsembleTest(unittest.TestCase):
    def _pred(self, model, point):
        return pd.DataFrame({
            "model": model, "pair": "SGD/IDR", "origin_date": "2024-01-01",
            "target_date": "2024-01-02", "evaluation_split": "val", "horizon": 1,
            "current_rate": 10000.0, "actual_rate": 10100.0,
            "point_forecast": point, "q10": point - 50, "q50": point, "q90": point + 50,
        }, index=[0])

    def test_weighted_combine_and_disagreement(self):
        per_model = {"a": self._pred("a", 10100.0), "b": self._pred("b", 10300.0)}
        weights = {"SGD/IDR": {"a": 0.75, "b": 0.25}}
        out = combine_predictions(per_model, weights, ("a", "b"))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out["point_forecast"].iloc[0], 0.75 * 10100 + 0.25 * 10300)
        self.assertAlmostEqual(out["disagreement"].iloc[0], float(np.std([10100.0, 10300.0])))


if __name__ == "__main__":
    unittest.main()
