import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from fastapi.middleware.cors import CORSMiddleware

from app.config import FRAUD_MODEL_VERSION
from app.fraud import service as fraud_service
from app.fx import advisor as fx_advisor
from app.schemas.fraud import FraudScoreRequest, FraudScoreResponse
from app.schemas.fx import FxAdvisoryRequest, FxAdvisoryResponse

logger = logging.getLogger("nusawallet.fraud")

# Populated at startup with the trained ensemble scorer, or left None so the service
# falls back to the demo model (e.g. when ML deps or artifacts are unavailable).
_scorer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer
    try:
        # Imported lazily so the API still starts without the heavy ML deps installed.
        from app.fraud.inference import FraudScorer

        _scorer = FraudScorer.load()  # loads artifacts; never trains
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Trained fraud scorer unavailable (%s); using demo model", exc)
        _scorer = None
    if _scorer is None:
        fraud_service.warm_up()  # prepare the lightweight demo fallback
    yield


app = FastAPI(
    title="NusaWallet AI Service",
    description="Intelligence Layer — FX Decision-Support (time-series) and fraud/anomaly scoring.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "nusawallet-ai", "fraud_model_loaded": _scorer is not None}


@app.get("/fx/advisory", response_model=FxAdvisoryResponse, tags=["fx"])
def fx_advisory(request: FxAdvisoryRequest = Depends()):
    return fx_advisor.advise(
        request.base, request.quote, request.amount,
        request.horizon_days, request.risk_preference.value,
    )


@app.post("/fraud/score", response_model=FraudScoreResponse, tags=["fraud"])
def fraud_score(request: FraudScoreRequest):
    if _scorer is not None:
        return _scorer.score(request)
    # Demo fallback keeps the endpoint working without the trained bundle.
    return fraud_service.score(
        request.amount,
        request.currency,
        request.payer_name,
        request.effective_hour,
    )


@app.get("/models/fraud/info", tags=["fraud"])
def fraud_model_info():
    if _scorer is not None:
        return _scorer.info()
    return {
        "available": False,
        "model": "IsolationForest + rules (demo)",
        "model_version": FRAUD_MODEL_VERSION,
    }
