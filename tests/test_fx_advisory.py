"""Phase 13 tests: the FX advisory endpoint serving the decision engine.

fetch_series is patched so the tests are deterministic and offline.
"""

import unittest
from unittest.mock import patch

from app.fx.advisor import advise
from app.schemas.fx import FxAdvisoryResponse

_RISING = [12000 * (1.003 ** i) for i in range(60)]   # ~0.3%/day up
_FALLING = [12000 * (0.997 ** i) for i in range(60)]  # ~0.3%/day down


class FxAdvisoryServingTest(unittest.TestCase):
    @patch("app.fx.advisor.fetch_series")
    def test_full_response_validates_and_is_populated(self, mock_fetch):
        mock_fetch.return_value = _RISING
        r = advise("SGD", "IDR", amount=1000, horizon_days=7, risk_preference="MODERATE")
        FxAdvisoryResponse.model_validate(r)  # raises on contract mismatch
        self.assertIsNotNone(r["forecast_rate"])
        self.assertIsNotNone(r["estimated_gain_loss"])
        self.assertIsNotNone(r["recommended_convert_percentage"])
        self.assertIn(r["action"], {"CONVERT_NOW", "HOLD_TEMPORARILY", "SPLIT_CONVERSION"})
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)
        self.assertEqual(r["pair"], "SGD/IDR")

    @patch("app.fx.advisor.fetch_series")
    def test_rising_series_holds(self, mock_fetch):
        mock_fetch.return_value = _RISING
        r = advise("SGD", "IDR", amount=1000, horizon_days=7)
        self.assertIn(r["action"], {"HOLD_TEMPORARILY", "SPLIT_CONVERSION"})
        self.assertLess(r["recommended_convert_percentage"], 100)

    @patch("app.fx.advisor.fetch_series")
    def test_falling_series_converts_now(self, mock_fetch):
        mock_fetch.return_value = _FALLING
        r = advise("SGD", "IDR", amount=1000, horizon_days=7)
        self.assertEqual(r["action"], "CONVERT_NOW")
        self.assertEqual(r["recommended_convert_percentage"], 100)

    @patch("app.fx.advisor.fetch_series", side_effect=RuntimeError("series down"))
    def test_fallback_to_legacy_on_failure(self, mock_fetch):
        r = advise("SGD", "IDR")
        FxAdvisoryResponse.model_validate(r)  # legacy response still valid
        self.assertIsNone(r["forecast_rate"])  # legacy leaves forecast null


if __name__ == "__main__":
    unittest.main()
