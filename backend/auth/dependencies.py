"""FastAPI dependency that resolves the current user from the JWT cookie.

Reads the httponly ``access_token`` cookie, verifies it, loads the user, and
raises 401 on any failure. Routes depend on ``get_current_user`` to require auth.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.auth.security import COOKIE_NAME, decode_access_token
from backend.database import get_db
from backend.db_models import User

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Return the authenticated User or raise 401."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise _UNAUTHENTICATED

    subject = decode_access_token(token)
    if subject is None:
        raise _UNAUTHENTICATED

    user = db.get(User, int(subject)) if subject.isdigit() else None
    if user is None:
        raise _UNAUTHENTICATED
    return user
