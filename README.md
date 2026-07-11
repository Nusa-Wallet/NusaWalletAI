# NusaWallet AI Service

> Continuing development? Read [HANDOFF.md](HANDOFF.md) for the agreed model,
> data, Kaggle, artifact, and next-phase plan. API compatibility is defined in
> [CONTRACTS.md](CONTRACTS.md).

The **Intelligence Layer**, run as a separate FastAPI service (port 8001). Two
explainable models, called by the core backend:

- **FX Advisory** (`GET /fx/advisory?base=SGD&quote=IDR`) — a Decision-Support
  System over the recent rate series. Returns `action` (CONVERT_NOW / WAIT / HOLD),
  `confidence`, moving average, volatility, best/worst scenario bounds, and a
  human-readable `rationale`. **Not** a black-box predictor (proposal section 8).
- **Fraud Scoring** (`POST /fraud/score`) — IsolationForest anomaly model +
  transparent business rules. Returns `risk_score`, `flagged`, and the
  contributing `factors`.

## Run (dev)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Docs: http://localhost:8001/docs

Data sources: free [Frankfurter](https://www.frankfurter.app/) FX API (no key),
with a deterministic synthetic fallback so demos work offline.

## Structure

```
app/
  schemas/           frozen Pydantic request/response contracts
  fx/                data, features, decision, model/ensemble boundary, service
  fraud/             features, rules, model, ensemble, explanations, service
  main.py            FastAPI endpoints
data/                raw, processed, and synthetic dataset workspaces
artifacts/           persisted model artifacts (generated files are ignored)
scripts/             future data/training/evaluation entry points
tests/               contract and service tests
CONTRACTS.md         version-1 API contract and compatibility policy
```

The current models remain the original statistical/Isolation Forest demo models.
The expanded structure prepares later data, training, evaluation, and model-serving
phases without claiming that the advanced models already exist.
