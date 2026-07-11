import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from app.fraud import service as fraud_service
from app.fx import service as fx_service
from app.schemas.fraud import FraudScoreRequest, FraudScoreResponse
from app.schemas.fx import FxAdvisoryRequest, FxAdvisoryResponse


class ContractTests(unittest.TestCase):
    def test_fraud_request_is_backward_compatible(self):
        request = FraudScoreRequest(amount=400, currency="sgd", payer_name="John")
        self.assertEqual(request.currency, "SGD")
        self.assertEqual(request.effective_hour, 12)

    def test_occurred_at_is_authoritative_for_hour(self):
        request = FraudScoreRequest(
            amount=400,
            currency="SGD",
            hour=12,
            occurred_at=datetime(2026, 7, 11, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(request.effective_hour, 2)

    def test_invalid_financial_inputs_are_rejected(self):
        with self.assertRaises(ValidationError):
            FraudScoreRequest(amount=0, currency="SGD")
        with self.assertRaises(ValidationError):
            FxAdvisoryRequest(base="SGD", quote="SGD")

    def test_current_services_satisfy_frozen_responses(self):
        FraudScoreResponse.model_validate(fraud_service.score(400, "SGD", "John", 12))
        fx = fx_service.advise("SGD", "IDR")
        FxAdvisoryResponse.model_validate(fx)
        self.assertIsNone(fx["forecast_rate"])


if __name__ == "__main__":
    unittest.main()
