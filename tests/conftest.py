"""Pytest fixtures: a TestClient backed by a temporary SQLite database.

The real ``backend/finsentinel.db`` is never touched — we point ``DATABASE_URL``
at a throwaway temp file *before* importing the app (so the app's engine and
``init_db()`` use it), and additionally override the ``get_db`` dependency with a
test session factory. Tables are truncated before each test for isolation and
the temp DB is deleted when the session ends.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- point the app at a temp DB BEFORE importing it --------------------------
_fd, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="finsentinel_test_")
os.close(_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_DB_PATH).as_posix()}"
os.environ.pop("ENVIRONMENT", None)  # keep the production secret-guard inactive

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import backend.db_models  # noqa: F401,E402 - registers models on Base.metadata
from backend.database import Base, get_db  # noqa: E402
from backend.main import app  # noqa: E402

# A dedicated test engine on the same temp DB, used by the override + direct inserts.
_engine = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
Base.metadata.create_all(bind=_engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(autouse=True)
def _clean_db():
    """Empty every table before each test so cases don't bleed into each other."""
    with _engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    yield


@pytest.fixture
def client():
    """A fresh TestClient (its own cookie jar) per test."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    """A test DB session for setting up rows directly (bypassing the API)."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    # Dispose BOTH engines bound to the temp DB (the app's and the test's) so
    # Windows releases the file lock before we delete it.
    from backend.database import engine as app_engine

    app_engine.dispose()
    _engine.dispose()
    try:
        os.unlink(_DB_PATH)
    except OSError:
        pass
