"""History routes: list and fetch a user's saved analysis runs.

Every record is scoped to the authenticated user — a user can only ever see
their own history. The list endpoint returns lightweight summaries; the detail
endpoint returns the full stored AnalysisResponse payload.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.auth.schemas import HistoryDetail, HistoryItem
from backend.database import get_db
from backend.db_models import AnalysisHistory, User

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=list[HistoryItem], summary="List the current user's analysis history")
def list_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AnalysisHistory]:
    """Return the user's most recent analysis runs (newest first)."""
    return (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.user_id == current_user.id)
        .order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc())
        .limit(limit)
        .all()
    )


@router.get("/{history_id}", response_model=HistoryDetail, summary="Fetch one saved analysis in full")
def get_history(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisHistory:
    """Return a single record (full payload), 404 if it is not the user's."""
    record = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.id == history_id, AnalysisHistory.user_id == current_user.id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return record
