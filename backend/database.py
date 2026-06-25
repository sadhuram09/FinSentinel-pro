"""SQLAlchemy database setup (SQLite now, Postgres-ready later).

The engine targets a local SQLite file (``backend/finsentinel.db``) so the app
runs with zero infrastructure. Everything else — the ORM models, the session
factory, the ``get_db`` dependency — is written against generic SQLAlchemy, so
swapping ``DATABASE_URL`` to a Postgres DSN later needs no schema or code
changes here.

The ``check_same_thread`` connect-arg is SQLite-specific (FastAPI serves
requests across threads); it is applied only for SQLite URLs so a future
Postgres URL is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Absolute path keeps the DB at backend/finsentinel.db regardless of the working
# directory the app is launched from.
DB_PATH = Path(__file__).resolve().parent / "finsentinel.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


def get_db():
    """FastAPI dependency yielding a request-scoped session, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Import models first so they register on ``Base``."""
    from backend import db_models  # noqa: F401 - registers models on Base.metadata

    Base.metadata.create_all(bind=engine)
