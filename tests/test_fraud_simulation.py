"""Phase 3 tests: reproducibility, integrity, anomaly coverage, and no leakage.

Skipped automatically when the ML dependencies (pandas/pandera) are absent so the
base-interpreter contract tests still run via ``python -m unittest discover``.
"""

import unittest

try:
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from app.fraud.simulation import SimulationConfig, generate_dataset
    from app.fraud.simulation.config import ANOMALY_TYPES, NORMAL_LABEL
    from app.fraud.simulation.features import compute_historical_features
    from app.fraud.simulation.schema import CANONICAL_COLUMNS, validate

    HAS_SIM_DEPS = True
except ImportError:
    HAS_SIM_DEPS = False

from app.config import SUPPORTED_CURRENCIES


@unittest.skipUnless(HAS_SIM_DEPS, "pandas/pandera not installed in this interpreter")
class FraudSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig.small_sample()
        cls.df, cls.metadata = generate_dataset(cls.config)

    def test_deterministic_same_seed(self):
        df2, _ = generate_dataset(SimulationConfig.small_sample())
        assert_frame_equal(self.df, df2)

    def test_row_count_matches_config(self):
        self.assertEqual(len(self.df), self.config.n_transactions)

    def test_canonical_column_order(self):
        self.assertEqual(tuple(self.df.columns), CANONICAL_COLUMNS)

    def test_transaction_ids_unique(self):
        self.assertEqual(self.df["transaction_id"].nunique(), len(self.df))

    def test_amounts_positive(self):
        self.assertTrue((self.df["amount"] > 0).all())
        self.assertTrue((self.df["amount_idr"] > 0).all())

    def test_currencies_supported(self):
        self.assertTrue(set(self.df["currency"]).issubset(SUPPORTED_CURRENCIES))

    def test_timestamps_within_window(self):
        start, end = self.config.start_ts(), self.config.end_ts()
        self.assertTrue((self.df["occurred_at"] >= start).all())
        self.assertTrue((self.df["occurred_at"] <= end).all())

    def test_every_anomaly_type_represented(self):
        present = set(self.df["anomaly_type"])
        self.assertTrue(set(ANOMALY_TYPES).issubset(present))

    def test_is_anomaly_consistent_with_label(self):
        expected = self.df["anomaly_type"] != NORMAL_LABEL
        self.assertTrue((self.df["is_anomaly"] == expected).all())

    def test_anomaly_ratio_in_reasonable_band(self):
        ratio = self.df["is_anomaly"].mean()
        self.assertGreater(ratio, 0.02)
        self.assertLess(ratio, 0.15)

    def test_schema_validation_passes(self):
        validate(self.df, self.config)  # raises on failure

    def test_metadata_provenance(self):
        m = self.metadata
        self.assertEqual(m["seed"], self.config.seed)
        self.assertEqual(m["row_count"], len(self.df))
        self.assertEqual(m["unique_transaction_ids"], len(self.df))
        self.assertGreater(m["anomaly"]["count"], 0)
        self.assertEqual(set(m["anomaly"]["type_counts"]) - {NORMAL_LABEL}, set(ANOMALY_TYPES))

    # --- Leakage / feature-correctness on a controlled frame ---------------

    def _controlled_frame(self):
        base = pd.Timestamp("2025-07-01T08:00:00Z")
        offsets = [0, 100, 700]  # seconds
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

    def test_velocity_windowing_uses_past_only(self):
        feats = compute_historical_features(self._controlled_frame())
        # First transaction: no history.
        self.assertEqual(feats["user_velocity_24h"].iloc[0], 0)
        self.assertEqual(feats["payer_velocity_10m"].iloc[0], 0)
        self.assertTrue(feats["is_new_payer"].iloc[0])
        self.assertFalse(feats["country_seen_before"].iloc[0])
        self.assertEqual(feats["amount_ratio_user"].iloc[0], 1.0)
        # Second (+100s): one prior txn, within 10m.
        self.assertEqual(feats["user_velocity_24h"].iloc[1], 1)
        self.assertEqual(feats["payer_velocity_10m"].iloc[1], 1)
        self.assertFalse(feats["is_new_payer"].iloc[1])
        self.assertTrue(feats["country_seen_before"].iloc[1])
        # Third (+700s): row0 now older than 10m, only row1 remains in window.
        self.assertEqual(feats["payer_velocity_10m"].iloc[2], 1)
        self.assertEqual(feats["user_velocity_24h"].iloc[2], 2)

    def test_features_invariant_to_future_rows(self):
        full = self._controlled_frame()
        feats_full = compute_historical_features(full)
        feats_prefix = compute_historical_features(full.iloc[:2].copy())
        assert_frame_equal(
            feats_full.iloc[:2].reset_index(drop=True),
            feats_prefix.reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
