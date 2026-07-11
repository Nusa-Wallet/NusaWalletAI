"""Phase 4 tests: canonical feature transforms and training/inference parity.

Stateless transforms and the shared ``build_features`` need no ML deps; the parity
test against the batch pass needs pandas, so it self-skips when pandas is absent.
"""

import unittest

from app.fraud.feature_spec import (
    FEATURE_GROUPS,
    MODEL_FEATURES,
    MISSING_DEFAULTS,
)
from app.fraud.features import (
    HistoricalContext,
    RawTransaction,
    build_features,
    duplicate_similarity,
    hour_cos,
    hour_sin,
    payer_name_quality,
)

try:
    import pandas as pd

    from app.fraud.simulation.features import compute_historical_features

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class FeatureSpecTest(unittest.TestCase):
    def test_flat_list_matches_groups(self):
        flat = tuple(n for g in FEATURE_GROUPS.values() for n in g)
        self.assertEqual(MODEL_FEATURES, flat)

    def test_no_duplicate_feature_names(self):
        self.assertEqual(len(MODEL_FEATURES), len(set(MODEL_FEATURES)))

    def test_missing_defaults_are_known_features(self):
        self.assertTrue(set(MISSING_DEFAULTS).issubset(set(MODEL_FEATURES)))


class StatelessTransformTest(unittest.TestCase):
    def test_hour_cyclical_encoding(self):
        self.assertAlmostEqual(hour_sin(0), 0.0)
        self.assertAlmostEqual(hour_cos(0), 1.0)
        self.assertAlmostEqual(hour_sin(6), 1.0)
        self.assertAlmostEqual(hour_cos(12), -1.0)
        # Midnight wrap-around: hour 23 is close to hour 0.
        self.assertLess(abs(hour_sin(23) - hour_sin(0)), 0.3)

    def test_payer_name_quality_good_vs_bad(self):
        self.assertGreater(payer_name_quality("John Tan"), 0.8)
        self.assertEqual(payer_name_quality(""), 0.0)
        self.assertEqual(payer_name_quality("x"), 0.0)
        self.assertLessEqual(payer_name_quality("n/a"), 0.1)
        self.assertLessEqual(payer_name_quality("zzz"), 0.1)
        self.assertLessEqual(payer_name_quality("Unknown"), 0.1)
        # Digits drag the score down relative to a clean name.
        self.assertLess(payer_name_quality("A1B2C3"), payer_name_quality("John Tan"))

    def test_duplicate_similarity(self):
        # Identical amount + same payer within window -> ~1.0
        recent = ((2_000_000.0, "p1", 120.0),)
        self.assertAlmostEqual(duplicate_similarity(2_000_000.0, "p1", recent), 1.0)
        # Same amount, different payer -> reduced by payer factor.
        self.assertAlmostEqual(duplicate_similarity(2_000_000.0, "p2", recent), 0.6)
        # No history -> 0.0
        self.assertEqual(duplicate_similarity(2_000_000.0, "p1", ()), 0.0)


class BuildFeaturesTest(unittest.TestCase):
    def test_first_transaction_uses_missing_defaults(self):
        raw = RawTransaction(amount_idr=5_000_000.0, payer_id="p1", payer_name="John Tan", hour=10, day_of_week=2)
        feats = build_features(raw, HistoricalContext())
        self.assertEqual(tuple(feats), MODEL_FEATURES)  # order preserved
        self.assertEqual(feats["amount_ratio_user"], MISSING_DEFAULTS["amount_ratio_user"])
        self.assertEqual(feats["amount_zscore_user"], MISSING_DEFAULTS["amount_zscore_user"])
        self.assertEqual(feats["duplicate_similarity"], 0.0)
        self.assertTrue(feats["is_new_payer"])
        self.assertEqual(feats["user_velocity_24h"], 0)

    def test_ratio_and_zscore_from_context(self):
        raw = RawTransaction(amount_idr=3_000_000.0, payer_id="p1", payer_name="John Tan", hour=8, day_of_week=1)
        ctx = HistoricalContext(user_txn_count=2, user_amount_mean_idr=1_500_000.0, user_amount_std_idr=500_000.0)
        feats = build_features(raw, ctx)
        self.assertAlmostEqual(feats["amount_ratio_user"], 2.0)
        self.assertAlmostEqual(feats["amount_zscore_user"], 3.0)


@unittest.skipUnless(HAS_PANDAS, "pandas not installed in this interpreter")
class TrainingInferenceParityTest(unittest.TestCase):
    """The batch pass must produce the same vector as a hand-built context."""

    def _frame(self):
        base = pd.Timestamp("2025-07-01T08:00:00Z")
        offsets = [0, 100, 700]
        return pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "payer_id": ["p1", "p1", "p1"],
                "payer_name": ["John Tan", "John Tan", "John Tan"],
                "amount_idr": [1_000_000.0, 2_000_000.0, 3_000_000.0],
                "currency": ["USD", "USD", "USD"],
                "origin_country": ["US", "US", "US"],
                "occurred_at": [base + pd.Timedelta(seconds=s) for s in offsets],
            }
        )

    def test_batch_row_matches_build_features(self):
        frame = self._frame()
        feats_frame = compute_historical_features(frame)

        # Independently reconstruct the context the third row should see (past-only).
        raw2 = RawTransaction(
            amount_idr=3_000_000.0, payer_id="p1", payer_name="John Tan",
            hour=8, day_of_week=1,
        )
        ctx2 = HistoricalContext(
            user_txn_count=2,
            user_amount_mean_idr=1_500_000.0,
            user_amount_std_idr=500_000.0,   # population std of [1e6, 2e6]
            is_new_payer=False,
            payer_seen_before=True,
            payer_age_days=0,
            payer_velocity_10m=1,            # only the +100s txn is within 10m of +700s
            payer_velocity_24h=2,
            user_velocity_24h=2,
            country_seen_before=True,
            currency_seen_before=True,
            recent_user_txns=((1_000_000.0, "p1", 700.0), (2_000_000.0, "p1", 600.0)),
        )
        expected = build_features(raw2, ctx2)

        for name in MODEL_FEATURES:
            if name == "amount_idr":
                continue  # kept as a raw column, not in the feature frame
            self.assertEqual(
                feats_frame[name].iloc[2], expected[name],
                msg=f"mismatch on feature {name!r}",
            )


if __name__ == "__main__":
    unittest.main()
