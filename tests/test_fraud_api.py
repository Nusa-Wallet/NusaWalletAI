"""Phase 7 tests: the trained fraud scorer served through FastAPI.

Skipped when the ML deps or trained artifacts are unavailable (the demo fallback path
is exercised by the existing contract tests).
"""

import unittest

from fastapi.testclient import TestClient

from app.main import app

try:
    from app.fraud.inference import DEFAULT_ARTIFACTS_DIR, FraudScorer
    from app.fraud.feature_spec import MODEL_FEATURES
    from app.fraud.training.pipeline import predict_risk
    from app.schemas.fraud import FraudScoreRequest

    ARTIFACTS_READY = (DEFAULT_ARTIFACTS_DIR / "fraud_catboost.cbm").exists()
except ImportError:
    ARTIFACTS_READY = False


@unittest.skipUnless(ARTIFACTS_READY, "trained fraud artifacts / ML deps unavailable")
class FraudApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()  # runs lifespan -> loads the trained scorer
        cls.scorer = FraudScorer.load()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_reports_model_loaded(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["fraud_model_loaded"])

    def test_minimal_request_is_backward_compatible(self):
        r = self.client.post("/fraud/score", json={"amount": 100, "currency": "USD"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("risk_score", "risk_level", "flagged", "recommended_action",
                    "factors", "component_scores", "model", "model_version"):
            self.assertIn(key, body)
        self.assertIn(body["risk_level"], {"LOW", "MEDIUM", "HIGH"})
        # trained model populates the supervised component (demo left it null)
        self.assertIsNotNone(body["component_scores"]["supervised"])
        self.assertTrue(body["model_version"].startswith("fraud-ensemble"))

    def test_high_risk_request_is_flagged_with_reasons(self):
        r = self.client.post("/fraud/score", json={
            "amount": 999999, "currency": "USD", "payer_name": "",
            "origin_country": "KP", "hour": 2, "is_new_payer": True,
            "transactions_last_10m": 8, "transactions_last_24h": 20,
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertNotEqual(body["risk_level"], "LOW")
        self.assertGreaterEqual(len(body["factors"]), 1)

    def test_invalid_inputs_rejected(self):
        self.assertEqual(self.client.post("/fraud/score", json={"amount": -5, "currency": "USD"}).status_code, 422)
        self.assertEqual(self.client.post("/fraud/score", json={"amount": 100, "currency": "XXX"}).status_code, 422)

    def test_model_info(self):
        body = self.client.get("/models/fraud/info").json()
        self.assertTrue(body["available"])
        self.assertEqual(body["feature_names"], list(MODEL_FEATURES))
        self.assertIsNotNone(body["test_metrics"])
        self.assertIn("high", body["thresholds"])

    def test_online_scoring_matches_shared_offline_path(self):
        # DoD: identical feature inputs -> identical result as the offline pipeline.
        req = FraudScoreRequest(
            amount=1500, currency="SGD", payer_name="John Doe", hour=2,
            is_new_payer=True, transactions_last_10m=1, transactions_last_24h=4,
        )
        api_risk = self.scorer.score(req)["risk_score"]
        frame = self.scorer._build_frame(
            amount=1500, currency="SGD", payer_name="John Doe",
            hour=req.effective_hour, day_of_week=0, origin_country=None,
            is_new_payer=True, tx_10m=1, tx_24h=4,
        )
        offline_risk = float(
            predict_risk(self.scorer.model, self.scorer.isolation, self.scorer.ensemble, frame)[0]
        )
        self.assertAlmostEqual(api_risk, round(offline_risk, 4), places=4)


if __name__ == "__main__":
    unittest.main()
