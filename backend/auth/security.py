"""Password hashing (passlib/bcrypt) and JWT issue/verify (PyJWT).

Kept free of FastAPI and DB imports so it is trivially unit-testable and reusable.
The signing secret comes from ``JWT_SECRET_KEY``; the in-code default exists only
so the app boots in development — it must be overridden in any real deployment.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# The insecure default used only so the app boots in local development. The
# startup guard in main.py refuses to run with this value in production.
DEV_SECRET_PLACEHOLDER = "dev-insecure-change-me"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", DEV_SECRET_PLACEHOLDER)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
# Cookie name the token is stored in (httponly), read back by the auth dependency.
COOKIE_NAME = "access_token"


def is_using_dev_secret() -> bool:
    """True if the JWT signing key is still the insecure development placeholder."""
    return JWT_SECRET_KEY == DEV_SECRET_PLACEHOLDER


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of a plaintext password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plaintext password against its stored hash."""
    return _pwd_context.verify(plain, hashed)


def create_access_token(subject: str | int) -> str:
    """Mint a signed JWT whose ``sub`` is the user id, with an expiry claim."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(subject), "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Return the token's subject (user id) if valid, else None.

    Returns None for any failure mode — bad signature, expiry, malformed token —
    so callers handle "not authenticated" uniformly.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")
