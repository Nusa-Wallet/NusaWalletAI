from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.fraud import service as fraud_service
from app.fx import service as fx_service
from app.schemas.fraud import FraudScoreRequest, FraudScoreResponse
from app.schemas.fx import FxAdvisoryRequest, FxAdvisoryResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    fraud_service.warm_up()  # temporary demo model; persisted artifacts come later
    yield


app = FastAPI(
    title="NusaWallet AI Service",
    description="Intelligence Layer — FX Decision-Support (time-series) and fraud/anomaly scoring.",
    version="0.2.0",
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
    return {"status": "ok", "service": "nusawallet-ai"}


@app.get("/fx/advisory", response_model=FxAdvisoryResponse, tags=["fx"])
def fx_advisory(request: FxAdvisoryRequest = Depends()):
    return fx_service.advise(request.base, request.quote)


@app.post("/fraud/score", response_model=FraudScoreResponse, tags=["fraud"])
def fraud_score(request: FraudScoreRequest):
    return fraud_service.score(
        request.amount,
        request.currency,
        request.payer_name,
        request.effective_hour,
    )
