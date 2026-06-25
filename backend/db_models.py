"""SQLAlchemy ORM models: User and AnalysisHistory.

Deliberately database-agnostic — only portable column types are used so the
same models run on SQLite today and Postgres later with no schema change:

  * ``Integer`` autoincrement primary keys (portable; swap to UUID later only if
    needed).
  * ``JSON`` for the stored analysis payload — SQLAlchemy maps this to TEXT on
    SQLite and native JSON/JSONB on Postgres transparently.
  * Timezone-aware ``DateTime`` defaulted in Python to UTC (SQLite has no native
    tz type, so we never rely on DB-side tz handling).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """An authenticated account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    history: Mapped[list["AnalysisHistory"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class AnalysisHistory(Base):
    """A persisted record of one /analysis/run for a user."""

    __tablename__ = "analysis_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), index=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    lookback_period: Mapped[str] = mapped_column(String(8), nullable=False)
    prediction: Mapped[str] = mapped_column(String(8), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    # Full AnalysisResponse payload (JSON on Postgres, TEXT on SQLite).
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="history")
