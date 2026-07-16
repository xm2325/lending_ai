from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas import CreditRiskResponse, CreditSequenceRequest

app = FastAPI(title="Lending AI Lab", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=CreditRiskResponse)
def score(_: CreditSequenceRequest) -> CreditRiskResponse:
    raise HTTPException(
        status_code=503,
        detail="No approved model artifact is loaded; scoring fails closed.",
    )
