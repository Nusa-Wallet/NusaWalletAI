from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.fraud import scorer
from app.fx import advisor


@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer.train()  # warm the anomaly model at startup
    yield


app = FastAPI(
    title="NusaWallet AI Service",
    description="Intelligence Layer — FX Decision-Support (time-series) and fraud/anomaly scoring.",
    version="0.1.0",
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


@app.get("/fx/advisory", tags=["fx"])
def fx_advisory(base: str = "SGD", quote: str = "IDR"):
    return advisor.advise(base.upper(), quote.upper())


class FraudRequest(BaseModel):
    amount: float
    currency: str
    payer_name: str = ""
    hour: int = 12


@app.post("/fraud/score", tags=["fraud"])
def fraud_score(req: FraudRequest):
    return scorer.score(req.amount, req.currency, req.payer_name, req.hour)
