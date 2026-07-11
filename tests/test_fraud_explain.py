"""Phase 6 tests: rule/SHAP factor generation, ranking, gating, and SHAP summary.

The factor service needs pandas (rules use a DataFrame); the SHAP tests additionally
need catboost. Both self-skip when their deps are absent.
"""

import unittest

try:
    import pandas as pd  # noqa: F401

    from app.fraud.explain import explain_if_flagged, explain_transaction
    from app.fraud.explain.templates import FALLBACK_FACTOR

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from app.fraud.explain.shap_explain import global_shap_summary, shap_matrix
    from app.fraud.feature_spec import MODEL_FEATURES
    from app.fraud.simulation import SimulationConfig, generate_dataset
    from app.fraud.training.data import to_xy
    from app.fraud.training.models import train_catboost

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def make_row(**overrides):
    """A complete, low-risk transaction row; override fields to trigger rules."""
    row = {
        "amount_idr": 5_000_000.0, "amount_ratio_user": 1.0, "amount_zscore_user": 0.0,
        "hour": 12, "hour_sin": 0.0, "hour_cos": -1.0, "day_of_week": 2,
        "duplicate_similarity": 0.0, "payer_velocity_10m": 0, "payer_velocity_24h": 0,
        "user_velocity_24h": 0, "is_new_payer": False, "payer_seen_before": True,
        "payer_age_days": 100, "payer_name_quality": 1.0,
        "country_seen_before": True, "currency_seen_before": True,
        "currency": "SGD", "origin_country": "SG",
    }
    row.update(overrides)
    return row


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class FactorServiceTest(unittest.TestCase):
    def test_amount_rule_factor_matches_value(self):
        exp = explain_transaction(make_row(amount_ratio_user=8.0))
        self.assertTrue(exp.factors)
        self.assertIn("8.0x", exp.factors[0])
        self.assertIn("pola normal", exp.factors[0].lower())

    def test_high_risk_country_factor(self):
        exp = explain_transaction(make_row(origin_country="KP", country_seen_before=False))
        joined = " ".join(exp.factors).lower()
        self.assertIn("berisiko tinggi", joined)

    def test_factors_capped_at_max(self):
        row = make_row(
            amount_ratio_user=9.0, amount_zscore_user=6.0, hour=3,
            payer_name_quality=0.05, payer_velocity_10m=5, duplicate_similarity=0.99,
            payer_seen_before=False, origin_country="KP",
            country_seen_before=False, currency_seen_before=False,
        )
        exp = explain_transaction(row, max_factors=3)
        self.assertLessEqual(len(exp.factors), 3)
        self.assertGreaterEqual(len(exp.factors), 1)

    def test_topic_deduplicated(self):
        # Two amount rules + amount SHAP must collapse to a single amount factor.
        shap_row = {"amount_ratio_user": 2.0, "amount_zscore_user": 1.0}
        exp = explain_transaction(make_row(amount_ratio_user=9.0, amount_zscore_user=6.0), shap_row)
        topics = [d["topic"] for d in exp.details]
        self.assertEqual(topics.count("amount"), 1)

    def test_shap_only_factor_when_no_rule(self):
        # No rule fires, but SHAP flags velocity -> a model factor appears.
        exp = explain_transaction(make_row(), shap_row={"payer_velocity_24h": 3.0})
        self.assertTrue(exp.factors)
        self.assertEqual(exp.details[0]["source"], "model")

    def test_deterministic(self):
        row = make_row(amount_ratio_user=8.0, payer_velocity_10m=4)
        self.assertEqual(explain_transaction(row).factors, explain_transaction(row).factors)

    def test_gating_below_threshold_is_empty(self):
        exp = explain_if_flagged(0.1, 0.5, make_row(amount_ratio_user=8.0))
        self.assertEqual(exp.factors, [])

    def test_flagged_always_has_reason(self):
        # Flagged but nothing specific triggers -> fallback reason.
        exp = explain_if_flagged(0.9, 0.5, make_row())
        self.assertEqual(exp.factors, [FALLBACK_FACTOR])


@unittest.skipUnless(HAS_CATBOOST, "catboost not installed")
class ShapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, _ = generate_dataset(SimulationConfig.small_sample(n_transactions=3000, months=4))
        cls.x, cls.y = to_xy(df)
        n = int(len(df) * 0.8)
        cls.model = train_catboost(cls.x[:n], cls.y[:n], cls.x[n:], cls.y[n:])

    def test_global_summary_covers_all_features(self):
        summary = global_shap_summary(self.model, self.x.head(500))
        self.assertEqual(set(summary), set(MODEL_FEATURES))
        self.assertTrue(all(v >= 0 for v in summary.values()))
        # sorted descending
        vals = list(summary.values())
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_shap_matrix_shape(self):
        m = shap_matrix(self.model, self.x.head(20))
        self.assertEqual(m.shape, (20, len(MODEL_FEATURES)))


if __name__ == "__main__":
    unittest.main()
