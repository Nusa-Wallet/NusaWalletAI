"""Reproducible NusaWallet fraud transaction simulator (Phase 3).

Generates a labelled synthetic cross-border transaction dataset with realistic
per-user behaviour and named, overlapping anomaly scenarios. Historical/behavioural
features are computed with past-data only to avoid label or future leakage.

Business logic lives here; ``scripts/generate_fraud_data.py`` only parses arguments
and calls :func:`generate_dataset`.
"""

from app.fraud.simulation.config import SimulationConfig
from app.fraud.simulation.generator import generate_dataset

__all__ = ["SimulationConfig", "generate_dataset"]
