# NusaWallet AI API Contracts

This document freezes the intended version-1 contracts before the advanced models
are implemented. Existing unversioned endpoints remain backward compatible while
the backend and mobile clients migrate.

## Compatibility policy

- `GET /fx/advisory` and `POST /fraud/score` remain available.
- New request fields are optional until the core backend supplies them.
- Existing response fields are not removed during the MVP.
- Model-derived fields that are not implemented yet remain `null`; the service must
  not fabricate forecasts or explanations.
- A future breaking contract must use a versioned route such as `/v2/...`.

## Fraud scoring

### Request

```json
{
  "transaction_id": "trx-001",
  "user_id": 42,
  "payer_id": "payer-7",
  "amount": 1500,
  "currency": "SGD",
  "payer_name": "John Doe",
  "origin_country": "SG",
  "occurred_at": "2026-07-11T02:30:00Z",
  "hour": 2,
  "is_new_payer": true,
  "transactions_last_10m": 1,
  "transactions_last_24h": 4
}
```

Only `amount` and `currency` are required for backward compatibility. When both
`occurred_at` and `hour` are supplied, `occurred_at` is authoritative.

### Response

```json
{
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "flagged": true,
  "recommended_action": "REVIEW_REQUIRED",
  "factors": ["Nominal jauh di atas pola transaksi normal."],
  "component_scores": {
    "supervised": null,
    "anomaly": 0.74,
    "rules": 0.8
  },
  "model": "IsolationForest + rules",
  "model_version": "fraud-baseline-0.1.0"
}
```

Risk levels are `LOW`, `MEDIUM`, and `HIGH`. Recommended actions are `ALLOW`,
`REVIEW_IF_NEEDED`, and `REVIEW_REQUIRED`. A score is decision support and does
not independently settle or reject a payment.

## FX advisory

### Current GET request

```http
GET /fx/advisory?base=SGD&quote=IDR
```

### Target request fields

```json
{
  "base": "SGD",
  "quote": "IDR",
  "amount": 1000,
  "horizon_days": 7,
  "risk_preference": "MODERATE"
}
```

### Response

```json
{
  "pair": "SGD/IDR",
  "action": "SPLIT_CONVERSION",
  "confidence": 0.78,
  "current_rate": 12750,
  "forecast_rate": 12820,
  "forecast_lower": 12500,
  "forecast_upper": 13050,
  "recommended_convert_percentage": 50,
  "estimated_gain_loss": 70000,
  "scenario_best": 13050,
  "scenario_worst": 12500,
  "rationale": "Model tidak sepakat dan volatilitas meningkat.",
  "reasons": ["Ketidakpastian prediksi cukup tinggi."],
  "model_version": "fx-1.0.0"
}
```

Target actions are `CONVERT_NOW`, `HOLD_TEMPORARILY`, and `SPLIT_CONVERSION`.
Legacy `WAIT` and `HOLD` values remain accepted while clients migrate. Forecast,
gain/loss, and split fields remain `null` until a validated forecasting model and
fee-aware decision engine are implemented.

FX output is an explainable scenario estimate, not a guarantee of profit or
regulated financial advice.

