"""Analysis router: the public HTTP surface for running a full analysis.

This layer is intentionally thin — validate input, delegate to the
orchestrator, translate domain errors into HTTP status codes. All business
logic lives in the agents/orchestrator so it stays framework-agnostic.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.agents.orchestrator import run_analysis
from backend.schemas import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/run", response_model=AnalysisResponse, summary="Run full ticker analysis")
def run(request: AnalysisRequest) -> AnalysisResponse:
    """Run the LangGraph pipeline (analyst -> forecast -> judge) for a ticker.

    Returns a payload containing the headline ``prediction``,
    ``direction_probability``, ``confidence_interval``, ``shap_explanation`` and
    ``model_version``, alongside the full analyst and forecast reports and the
    judge's adjudication (``judge_verdict``).
    """
    try:
        return run_analysis(request)
    except ValueError as exc:
        # Bad ticker / no data — a client error.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures cleanly
        logger.exception("Analysis failed for ticker %s", request.ticker)
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc
