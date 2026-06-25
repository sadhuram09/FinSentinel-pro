"""FinSentinel FastAPI application entry point.

Wires the routers together and exposes a health check. Run locally with:

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI

from backend.routers import analysis, mlops

# Load environment variables from .env before anything reads config.
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="FinSentinel",
    description=(
        "Agentic equity analysis: a descriptive analyst agent and a predictive "
        "forecast agent (stacked XGBoost + LightGBM ensemble) with SHAP "
        "explanations."
    ),
    version="1.0.0",
)

app.include_router(analysis.router)
app.include_router(mlops.router)


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> dict[str, str]:
    """Return a simple OK so orchestrators can verify the service is up."""
    return {"status": "ok"}
