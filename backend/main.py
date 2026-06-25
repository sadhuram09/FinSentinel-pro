"""FinSentinel FastAPI application entry point.

Wires the routers together and exposes a health check. Run locally with:

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import router as auth_router
from backend.auth.security import is_using_dev_secret
from backend.database import init_db
from backend.routers import analysis, history, mlops

# Load environment variables from .env before anything reads config.
load_dotenv()

logging.basicConfig(level=logging.INFO)


def _guard_production_secret() -> None:
    """Refuse to boot with the insecure dev JWT secret in a production environment.

    A leaked/default signing key lets anyone forge auth cookies, so shipping the
    placeholder is a critical misconfiguration. We fail loudly at startup rather
    than silently serving with it.
    """
    if os.getenv("ENVIRONMENT", "development").lower() == "production" and is_using_dev_secret():
        raise RuntimeError(
            "Refusing to start: JWT_SECRET_KEY is still the insecure development "
            "default while ENVIRONMENT=production. Set JWT_SECRET_KEY to a real "
            'random secret, e.g. python -c "import secrets; print(secrets.token_hex(32))".'
        )


_guard_production_secret()

# Create database tables (users, analysis_history) if they don't exist.
init_db()

app = FastAPI(
    title="FinSentinel",
    description=(
        "Agentic equity analysis: a descriptive analyst agent and a predictive "
        "forecast agent (stacked XGBoost + LightGBM ensemble) with SHAP "
        "explanations."
    ),
    version="1.0.0",
)

# Allow the Vite dev frontend to call the API *with credentials* (the httpOnly
# session cookie). allow_credentials=True requires explicit origins, never "*".
# Override allowed origins via CORS_ORIGINS (comma-separated) in deployment.
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5180").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(analysis.router)
app.include_router(history.router)
app.include_router(mlops.router)


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> dict[str, str]:
    """Return a simple OK so orchestrators can verify the service is up."""
    return {"status": "ok"}
