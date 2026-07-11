from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.config import SUPPORTED_CURRENCIES


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(StrEnum):
    ALLOW = "ALLOW"
    REVIEW_IF_NEEDED = "REVIEW_IF_NEEDED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class FraudScoreRequest(BaseModel):
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    payer_name: str = Field(default="", max_length=200)
    hour: int = Field(default=12, ge=0, le=23)
    transaction_id: str | None = Field(default=None, max_length=100)
    user_id: int | None = Field(default=None, gt=0)
    payer_id: str | None = Field(default=None, max_length=100)
    origin_country: str | None = Field(default=None, min_length=2, max_length=2)
    occurred_at: datetime | None = None
    is_new_payer: bool | None = None
    transactions_last_10m: int | None = Field(default=None, ge=0)
    transactions_last_24h: int | None = Field(default=None, ge=0)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        currency = value.upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}")
        return currency

    @field_validator("origin_country")
    @classmethod
    def normalize_country(cls, value: str | None) -> str | None:
        return value.upper() if value else None

    @property
    def effective_hour(self) -> int:
        return self.occurred_at.hour if self.occurred_at else self.hour


class FraudComponentScores(BaseModel):
    supervised: float | None = Field(default=None, ge=0, le=1)
    anomaly: float | None = Field(default=None, ge=0, le=1)
    rules: float | None = Field(default=None, ge=0, le=1)


class FraudScoreResponse(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    flagged: bool
    recommended_action: RecommendedAction
    factors: list[str]
    component_scores: FraudComponentScores
    model: str
    model_version: str
