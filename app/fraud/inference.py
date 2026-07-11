"""Online fraud scoring with the trained ensemble (Phase 7).

Loads the persisted artifacts once (never trains at startup) and scores a single
transaction by building its feature vector through the SAME
``app.fraud.features.build_features`` path used in training — so the API result equals
the offline result for identical feature inputs. Heavy ML deps are imported here (not
in main.py), so if they or the artifacts are missing the API falls back to the demo model.

Online context limitation: the request carries only partial history (velocity hints,
is_new_payer). Everything else uses the documented missing-value policy
(``HistoricalContext`` defaults); the backend should supply richer history when it can.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.fraud.explain import explain_transaction
from app.fraud.explain.shap_explain import shap_matrix, shap_row_dict
from app.fraud.features import HistoricalContext, RawTransaction, build_features
from app.fraud.feature_spec import MODEL_FEATURES
from app.fraud.rules_engine import rules_score
from app.fraud.simulation.config import RATE_TO_IDR
from app.fraud.training.data import feature_matrix
from app.fraud.training.models import catboost_proba
from app.fraud.training.pipeline import load_bundle

logger = logging.getLogger("nusawallet.fraud")

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_DESCRIPTION = "CatBoost + IsolationForest + rules (calibrated ensemble)"


@dataclass
class FraudScorer:
    model: object
    isolation: object
    ensemble: object
    metadata: dict
    high_threshold: float
    medium_threshold: float

    @classmethod
    def load(cls, artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR) -> "FraudScorer | None":
        """Load the trained bundle, or return None if artifacts are unavailable."""
        try:
            model, isolation, ensemble, metadata = load_bundle(Path(artifacts_dir))
        except Exception as exc:  # missing/corrupt artifacts -> caller uses demo fallback
            logger.warning("Fraud bundle unavailable (%s); demo fallback in effect", exc)
            return None
        high = float(ensemble.threshold)
        logger.info("Loaded fraud bundle %s (threshold=%.3f)", metadata.get("model_version"), high)
        return cls(model, isolation, ensemble, metadata, high, high / 2)

    def _build_frame(
        self, *, amount: float, currency: str, payer_name: str, hour: int, day_of_week: int,
        origin_country: str | None, is_new_payer: bool | None,
        tx_10m: int | None, tx_24h: int | None,
    ) -> pd.DataFrame:
        # Same conversion the model was trained on, so amount_idr is comparable.
        amount_idr = float(amount) * RATE_TO_IDR[currency]
        ctx = HistoricalContext(
            user_txn_count=0,  # no online history -> missing-value policy for amount stats
            is_new_payer=True if is_new_payer is None else bool(is_new_payer),
            payer_seen_before=False if is_new_payer is None else (not is_new_payer),
            payer_velocity_10m=int(tx_10m or 0),
            payer_velocity_24h=int(tx_24h or 0),
            user_velocity_24h=int(tx_24h or 0),
        )
        raw = RawTransaction(
            amount_idr=amount_idr, payer_id="",
            payer_name=payer_name or "", hour=int(hour), day_of_week=int(day_of_week),
        )
        features = build_features(raw, ctx)
        row = {**features, "hour": int(hour), "currency": currency,
               "origin_country": origin_country or "", "payer_name": payer_name or ""}
        return pd.DataFrame([row])

    def _level(self, risk: float) -> tuple[str, str, bool]:
        if risk >= self.high_threshold:
            return "HIGH", "REVIEW_REQUIRED", True
        if risk >= self.medium_threshold:
            return "MEDIUM", "REVIEW_IF_NEEDED", False
        return "LOW", "ALLOW", False

    def score(self, request) -> dict:
        day_of_week = request.occurred_at.weekday() if request.occurred_at else 0
        frame = self._build_frame(
            amount=request.amount, currency=request.currency, payer_name=request.payer_name,
            hour=request.effective_hour, day_of_week=day_of_week,
            origin_country=request.origin_country, is_new_payer=request.is_new_payer,
            tx_10m=request.transactions_last_10m, tx_24h=request.transactions_last_24h,
        )
        x = feature_matrix(frame)
        supervised = float(catboost_proba(self.model, x)[0])
        anomaly = float(self.isolation.score(x)[0])
        rules = float(rules_score(frame)[0])
        risk = float(self.ensemble.risk(
            np.array([supervised]), np.array([anomaly]), np.array([rules])
        )[0])

        level, action, flagged = self._level(risk)
        factors: list[str] = []
        if risk >= self.medium_threshold:
            shap_row = shap_row_dict(shap_matrix(self.model, x)[0], list(MODEL_FEATURES))
            factors = explain_transaction(
                frame.iloc[0].to_dict(), shap_row, ensure_reason=flagged
            ).factors

        logger.info(
            "fraud.score tx=%s level=%s risk=%.3f model=%s",
            request.transaction_id, level, risk, self.metadata.get("model_version"),
        )
        return {
            "risk_score": round(risk, 4),
            "risk_level": level,
            "flagged": flagged,
            "recommended_action": action,
            "factors": factors,
            "component_scores": {
                "supervised": round(supervised, 4),
                "anomaly": round(anomaly, 4),
                "rules": round(rules, 4),
            },
            "model": MODEL_DESCRIPTION,
            "model_version": self.metadata.get("model_version", "fraud-ensemble"),
        }

    def info(self) -> dict:
        m = self.metadata
        return {
            "available": True,
            "model": MODEL_DESCRIPTION,
            "model_version": m.get("model_version"),
            "dataset_version": m.get("dataset_version"),
            "schema_version": m.get("schema_version"),
            "trained_at": m.get("trained_at"),
            "git_commit": m.get("git_commit"),
            "feature_names": m.get("feature_names"),
            "ensemble_weights": m.get("ensemble_weights"),
            "thresholds": {"high": self.high_threshold, "medium": self.medium_threshold},
            "test_metrics": m.get("metrics", {}).get("full-ensemble"),
            "shap_global_summary_top": m.get("shap_global_summary_top"),
        }
