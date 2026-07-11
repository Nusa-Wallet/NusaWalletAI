"""Fraud model training pipeline (Phase 5).

Trains a calibrated CatBoost classifier plus an Isolation Forest and transparent
rules, combines them into a calibrated ensemble risk probability, tunes on a
chronological validation split, and evaluates once on a held-out test split. All
runnable on CPU for the local 50k dataset and portable to Kaggle GPU for the full run.
"""

from app.fraud.training.pipeline import TrainingConfig, run_training

__all__ = ["TrainingConfig", "run_training"]
