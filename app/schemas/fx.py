from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import SUPPORTED_CURRENCIES


class RiskPreference(StrEnum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class FxAction(StrEnum):
    CONVERT_NOW = "CONVERT_NOW"
    HOLD_TEMPORARILY = "HOLD_TEMPORARILY"
    SPLIT_CONVERSION = "SPLIT_CONVERSION"
    WAIT = "WAIT"
    HOLD = "HOLD"


class FxAdvisoryRequest(BaseModel):
    base: str = Field(default="SGD", min_length=3, max_length=3)
    quote: str = Field(default="IDR", min_length=3, max_length=3)
    amount: float | None = Field(default=None, gt=0)
    horizon_days: int = Field(default=7, ge=1, le=30)
    risk_preference: RiskPreference = RiskPreference.MODERATE

    @field_validator("base", "quote")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        currency = value.upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}")
        return currency

    @model_validator(mode="after")
    def currencies_must_differ(self):
        if self.base == self.quote:
            raise ValueError("Base and quote currencies must differ")
        return self


class FxAdvisoryResponse(BaseModel):
    pair: str
    action: FxAction
    confidence: float = Field(ge=0, le=1)
    current_rate: float = Field(gt=0)
    ma_7d: float | None = None
    volatility_7d: float | None = Field(default=None, ge=0)
    z_score: float | None = None
    forecast_rate: float | None = Field(default=None, gt=0)
    forecast_lower: float | None = Field(default=None, gt=0)
    forecast_upper: float | None = Field(default=None, gt=0)
    recommended_convert_percentage: int | None = Field(default=None, ge=0, le=100)
    estimated_gain_loss: float | None = None
    scenario_best: float
    scenario_worst: float
    rationale: str
    reasons: list[str] = Field(default_factory=list)
    model_version: str
