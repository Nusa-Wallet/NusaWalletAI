"""Phase 5 tests: chronological split, rules baseline, and an end-to-end pipeline run.

Skipped when the training deps (pandas/catboost) are unavailable so the base
interpreter's contract tests still run.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd

    from app.fraud.feature_spec import MODEL_FEATURES
    from app.fraud.simulation import SimulationConfig, generate_dataset
    from app.fraud.training.data import time_split, to_xy
    from app.fraud.training.models import rules_score
    from app.fraud.training.pipeline import TrainingConfig, load_bundle, predict_risk, run_training

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "pandas/catboost not installed in this interpreter")
class DataAndRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = generate_dataset(SimulationConfig.small_sample())

    def test_time_split_is_chronological_and_disjoint(self):
        s = time_split(self.df, 0.6, 0.2)
        self.assertEqual(len(s.train) + len(s.val) + len(s.test), len(self.df))
        self.assertLessEqual(s.train["occurred_at"].max(), s.val["occurred_at"].min())
        self.assertLessEqual(s.val["occurred_at"].max(), s.test["occurred_at"].min())

    def test_to_xy_columns_and_dtypes(self):
        x, y = to_xy(self.df)
        self.assertEqual(tuple(x.columns), MODEL_FEATURES)
        self.assertTrue(set(y.unique()).issubset({0, 1}))
        # bool features are coerced to int for the model
        self.assertEqual(x["is_new_payer"].dtype.kind, "i")

    def test_rules_score_in_unit_range(self):
        s = rules_score(self.df)
        self.assertGreaterEqual(s.min(), 0.0)
        self.assertLessEqual(s.max(), 1.0)


@unittest.skipUnless(HAS_DEPS, "pandas/catboost not installed in this interpreter")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        df, _ = generate_dataset(
            SimulationConfig.small_sample(n_transactions=4000, n_users=60, n_payers=200, months=4)
        )
        cls.dataset_path = root / "ds.parquet"
        df.to_parquet(cls.dataset_path, index=False)
        cls.artifacts = root / "artifacts"
        cls.config = TrainingConfig(
            dataset_path=str(cls.dataset_path),
            artifacts_dir=str(cls.artifacts),
            n_trials=2,
            mlflow_enabled=False,
        )
        cls.report = run_training(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_experiments_present(self):
        self.assertEqual(
            set(self.report["experiments"]),
            {"rules-only", "isolation-forest", "catboost-v1", "catboost-tuned", "full-ensemble"},
        )

    def test_metrics_in_valid_ranges(self):
        for name, m in self.report["experiments"].items():
            for key in ("precision", "recall", "f1", "fpr", "brier"):
                self.assertGreaterEqual(m[key], 0.0, f"{name}.{key}")
                self.assertLessEqual(m[key], 1.0, f"{name}.{key}")

    def test_ensemble_discriminates(self):
        ens = self.report["experiments"]["full-ensemble"]
        self.assertIsNotNone(ens["roc_auc"])
        self.assertGreater(ens["roc_auc"], 0.7)  # strong synthetic signal

    def test_definition_of_done_flags_present(self):
        dod = self.report["definition_of_done"]
        self.assertIn("fpr_below_target", dod)
        self.assertIn("beats_rules_f1", dod)

    def test_artifacts_saved(self):
        for fname in ("fraud_catboost.cbm", "fraud_isolation.joblib",
                      "fraud_calibrator.joblib", "fraud_metadata.json"):
            self.assertTrue((self.artifacts / fname).exists(), fname)

    def test_metadata_has_required_provenance(self):
        meta = self.report["metadata"]
        self.assertEqual(meta["feature_names"], list(MODEL_FEATURES))
        for key in ("model_version", "dataset_version", "schema_version", "threshold",
                    "ensemble_weights", "catboost_params", "library_versions"):
            self.assertIn(key, meta)

    def test_reloaded_bundle_predicts_deterministically(self):
        model, isolation, ensemble, _ = load_bundle(self.artifacts)
        df = pd.read_parquet(self.dataset_path).head(200)
        r1 = predict_risk(model, isolation, ensemble, df)
        r2 = predict_risk(model, isolation, ensemble, df)
        self.assertEqual(len(r1), len(df))
        self.assertTrue((r1 == r2).all())
        self.assertGreaterEqual(r1.min(), 0.0)
        self.assertLessEqual(r1.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
