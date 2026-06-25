"""Pydantic v2 DTOs for the auth and history APIs (separate from ORM models)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Signup payload."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128, description="Plaintext password (>= 8 chars).")


class UserLogin(BaseModel):
    """Login payload."""

    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Public view of a user (never exposes the password hash)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime


class HistoryItem(BaseModel):
    """Summary row for the history list."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    ticker: str
    lookback_period: str
    prediction: str
    verdict: str
    created_at: datetime


class HistoryDetail(HistoryItem):
    """A single history record including the full stored analysis payload."""

    result_json: dict
