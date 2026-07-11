"""Backward-compatible adapter for legacy imports."""

from app.fraud.service import score, warm_up

train = warm_up

__all__ = ["score", "train"]
