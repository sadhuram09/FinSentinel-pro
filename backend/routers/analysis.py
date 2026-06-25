"""Analysis router: the public HTTP surface for running a full analysis.

This layer is intentionally thin — require auth, validate input, delegate to the
orchestrator, persist the run to the user's history, and translate domain errors
into HTTP status codes. All business logic lives in the agents/orchestrator so it
stays framework-agnostic.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.agents.orchestrator import run_analysis
from backend.auth.dependencies import get_current_user
from backend.database import get_db
from backend.db_models import AnalysisHistory, User
from backend.schemas import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _save_history(
    db: Session, user_id: int, request: AnalysisRequest, response: AnalysisResponse
) -> int | None:
    """Persist a completed run to the user's history; return its id (best-effort)."""
    try:
        record = AnalysisHistory(
            user_id=user_id,
            ticker=response.ticker,
            lookback_period=request.lookback_period,
            prediction=response.prediction.value,
            verdict=response.judge_verdict.verdict.value,
            result_json=response.model_dump(mode="json"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    except Exception:  # noqa: BLE001 - a history write must not fail the analysis
        logger.exception("Failed to persist analysis history for user %s.", user_id)
        db.rollback()
        return None


@router.post("/run", response_model=AnalysisResponse, summary="Run full ticker analysis (auth required)")
def run(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisResponse:
    """Run the LangGraph pipeline for a ticker and save the result to history.

    Requires authentication. Returns the headline ``prediction``,
    ``direction_probability``, ``confidence_interval``, ``shap_explanation`` and
    ``model_version``, the full analyst/forecast/risk/sentiment reports, the RAG
    evidence, and the judge's adjudication (``judge_verdict``).
    """
    try:
        response = run_analysis(request)
    except ValueError as exc:
        # Bad ticker / no data — a client error.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures cleanly
        logger.exception("Analysis failed for ticker %s", request.ticker)
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc

    # Surface the saved record id so the client can navigate to /analysis/:id.
    response.id = _save_history(db, current_user.id, request, response)
    return response
