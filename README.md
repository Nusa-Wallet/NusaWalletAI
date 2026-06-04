# NusaWallet AI Service

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
  fx/advisor.py      time-series stats -> recommendation + confidence + rationale
  fraud/scorer.py    IsolationForest + rules -> risk score + factors
  main.py            FastAPI endpoints
```
