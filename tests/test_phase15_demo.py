"""Minimum offline AI demo scenarios required by Phase 15."""

import unittest
from fastapi.testclient import TestClient

from app.fx.decision.engine import decide
from app.main import app


class Phase15DemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def _fraud(self, payload):
        response = self.client.post("/fraud/score", json=payload)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_01_normal_transaction_is_low(self):
        result = self._fraud({"amount": 100, "currency": "USD", "payer_name": "John Doe"})
        self.assertEqual(result["risk_level"], "LOW")

    def test_02_large_new_payer_is_high(self):
        result = self._fraud({"amount": 999999, "currency": "USD", "payer_name": "",
                              "origin_country": "KP", "hour": 2, "is_new_payer": True})
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["recommended_action"], "REVIEW_REQUIRED")

    def test_03_velocity_burst_requires_review(self):
        result = self._fraud({"amount": 1000, "currency": "USD", "payer_name": "John Doe",
                              "origin_country": "US", "is_new_payer": False,
                              "transactions_last_10m": 12, "transactions_last_24h": 30})
        self.assertEqual(result["recommended_action"], "REVIEW_REQUIRED")

    def test_04_stable_fx_converts_now(self):
        result = decide(pair="SGD/IDR", current_rate=10000, forecast_rate=10000,
                        forecast_lower=9990, forecast_upper=10010, disagreement=0,
                        amount=1000, horizon_days=7, risk_preference="MODERATE")
        self.assertEqual(result.action, "CONVERT_NOW")

    def test_05_model_disagreement_splits_conversion(self):
        result = decide(pair="SGD/IDR", current_rate=10000, forecast_rate=10300,
                        forecast_lower=10050, forecast_upper=10550, disagreement=50,
                        amount=1000, horizon_days=7, risk_preference="MODERATE")
        self.assertEqual(result.action, "SPLIT_CONVERSION")
        self.assertTrue(0 < result.recommended_convert_percentage < 100)


if __name__ == "__main__":
    unittest.main()
