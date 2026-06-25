"""Auth routes: signup, login, logout, me — JWT delivered as an httponly cookie.

A cookie (not an Authorization header) is used so the browser stores the token
out of reach of JavaScript (httponly), which mitigates XSS token theft. Signup
and login both set the cookie, so the client is authenticated immediately.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.auth.schemas import UserCreate, UserLogin, UserOut
from backend.auth.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    COOKIE_NAME,
    create_access_token,
    hash_password,
    verify_password,
)
from backend.database import get_db
from backend.db_models import User

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_auth_cookie(response: Response, user_id: int) -> None:
    """Issue a fresh token and attach it as an httponly cookie."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=create_access_token(user_id),
        httponly=True,
        samesite="lax",
        secure=False,  # dev over http; set True behind HTTPS in production
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, response: Response, db: Session = Depends(get_db)) -> User:
    """Create an account and log the new user in (sets the auth cookie)."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    _set_auth_cookie(response, user.id)
    return user


@router.post("/login", response_model=UserOut)
def login(payload: UserLogin, response: Response, db: Session = Depends(get_db)) -> User:
    """Verify credentials and set the auth cookie."""
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        # Same message for unknown email vs wrong password (no user enumeration).
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    _set_auth_cookie(response, user.id)
    return user


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(response: Response) -> dict[str, str]:
    """Clear the auth cookie."""
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"detail": "Logged out"}


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user."""
    return current_user
