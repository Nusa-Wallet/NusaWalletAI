"""Phase 8 tests: cross-rate math, leak-free features, splits, walk-forward, build.

Runs fully offline using the deterministic synthetic provider (no network). Skipped when
pandas/pandera are unavailable.
"""

import unittest

try:
    import numpy as np
    import pandas as pd

    from app.fx.dataset.build import build_dataset
    from app.fx.dataset.config import FxDatasetConfig
    from app.fx.dataset.crossrates import compute_long_panel
    from app.fx.dataset.features import add_features
    from app.fx.dataset.providers import synthetic_eur_base
    from app.fx.dataset.schema import validate
    from app.fx.dataset.splits import assign_split, make_walk_forward_windows

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def _synthetic_fetcher(config):
    frame = synthetic_eur_base(config)
    provenance = {"provider": "synthetic-fallback", "n_observations": int(len(frame))}
    return frame, provenance


@unittest.skipUnless(HAS_DEPS, "pandas/pandera not installed")
class CrossRateTest(unittest.TestCase):
    def _eur_frame(self):
        idx = pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True)
        return pd.DataFrame({"EUR": [1.0, 1.0], "SGD": [1.5, 1.6], "IDR": [16000.0, 16500.0]}, index=idx)

    def test_cross_rate_formula(self):
        panel = compute_long_panel(self._eur_frame(), ["SGD/IDR", "EUR/IDR"])
        sgd_idr = panel[panel["pair"] == "SGD/IDR"].sort_values("date")["rate"].to_numpy()
        eur_idr = panel[panel["pair"] == "EUR/IDR"].sort_values("date")["rate"].to_numpy()
        np.testing.assert_allclose(sgd_idr, [16000 / 1.5, 16500 / 1.6])
        np.testing.assert_allclose(eur_idr, [16000.0, 16500.0])  # EUR base -> e[IDR]

    def test_reciprocal_pairs_multiply_to_one(self):
        panel = compute_long_panel(self._eur_frame(), ["SGD/IDR", "IDR/SGD"])
        a = panel[panel["pair"] == "SGD/IDR"].sort_values("date")["rate"].to_numpy()
        b = panel[panel["pair"] == "IDR/SGD"].sort_values("date")["rate"].to_numpy()
        np.testing.assert_allclose(a * b, [1.0, 1.0], rtol=1e-9)


@unittest.skipUnless(HAS_DEPS, "pandas/pandera not installed")
class FeatureLeakageTest(unittest.TestCase):
    def _panel(self):
        cfg = FxDatasetConfig.small_sample()
        panel = compute_long_panel(synthetic_eur_base(cfg), ["SGD/IDR"])
        return add_features(panel, cfg).reset_index(drop=True)

    def test_lag_is_previous_return(self):
        p = self._panel()
        # return_lag_1[t] must equal log_return[t-1] (past only).
        self.assertTrue(np.isnan(p["return_lag_1"].iloc[0]) or np.isnan(p["log_return"].iloc[0]))
        merged = p["return_lag_1"].iloc[2] == p["log_return"].iloc[1]
        self.assertTrue(merged)

    def test_features_invariant_to_future_rows(self):
        cfg = FxDatasetConfig.small_sample()
        panel = compute_long_panel(synthetic_eur_base(cfg), ["SGD/IDR"]).reset_index(drop=True)
        full = add_features(panel, cfg).reset_index(drop=True)
        prefix = add_features(panel.iloc[:50].copy(), cfg).reset_index(drop=True)
        cols = ["log_return", "return_lag_1", "roll_mean_5"]
        pd.testing.assert_frame_equal(full[cols].iloc[:50], prefix[cols], check_dtype=False)


@unittest.skipUnless(HAS_DEPS, "pandas/pandera not installed")
class SplitAndBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = FxDatasetConfig.small_sample()
        cls.panel, cls.meta, _, cls.prov = build_dataset(cls.cfg, verify=False, fetcher=_synthetic_fetcher)

    def test_split_is_chronological(self):
        by = {s: (g["date"].min(), g["date"].max()) for s, g in self.panel.groupby("split")}
        self.assertLess(by["train"][1], by["val"][0])
        self.assertLess(by["val"][1], by["test"][0])

    def test_no_duplicate_pair_date(self):
        self.assertFalse(self.panel.duplicated(["pair", "date"]).any())

    def test_schema_validates(self):
        validate(self.panel, self.cfg)

    def test_walk_forward_windows_ordered(self):
        windows = self.meta["walk_forward"]["windows"]
        self.assertGreater(len(windows), 0)
        for w in windows:
            self.assertLess(w["train_end"], w["test_start"])
            self.assertLessEqual(w["test_start"], w["test_end"])

    def test_metadata_records_provider_and_pairs(self):
        self.assertEqual(self.prov["provider"], "synthetic-fallback")
        self.assertEqual(self.meta["pairs"]["count"], len(self.cfg.pairs()))
        self.assertIn("SGD/IDR", self.meta["primary_rate_stats"])

    def test_reproducible_from_seed(self):
        panel2, _, _, _ = build_dataset(self.cfg, verify=False, fetcher=_synthetic_fetcher)
        pd.testing.assert_frame_equal(self.panel, panel2)


if __name__ == "__main__":
    unittest.main()
