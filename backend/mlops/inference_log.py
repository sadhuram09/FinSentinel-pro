"""SQLite log of inference-time feature vectors, for drift monitoring.

Every ``/analysis/run`` call records the technical feature row the model scored.
Persisting these lets the drift detector compare the *live* feature distribution
against the training baseline — the canonical way to catch a model going stale
because the world moved out from under its training data.

SQLite is deliberate: it is zero-ops, file-based and more than enough at this
scale. A later phase can swap in Postgres without touching callers, since all
DB access is funnelled through this module.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.ml.models.forecaster import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Project-root sqlite file (gitignored). inference_log.py is at backend/mlops/,
# so parents[2] is the project root.
DB_PATH = Path(__file__).resolve().parents[2] / "inference_log.db"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create the inference table if it does not exist (features stored as JSON)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            features_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def log_inference(ticker: str, features: dict) -> None:
    """Append one feature row for a scored request. Never raises into the caller.

    Args:
        ticker: The analysed ticker.
        features: Mapping of FEATURE_COLUMNS -> value (the row the model scored).
    """
    try:
        row = {k: features.get(k) for k in FEATURE_COLUMNS}
        with _connect() as conn:
            _ensure_table(conn)
            conn.execute(
                "INSERT INTO inference_log (ts, ticker, features_json) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), ticker.upper(), json.dumps(row)),
            )
            conn.commit()
    except Exception:  # noqa: BLE001 - logging must never break the request
        logger.exception("Failed to log inference features for %s.", ticker)


def load_recent_features(n: int = 100) -> pd.DataFrame:
    """Return the most recent ``n`` logged feature rows as a DataFrame.

    Columns are exactly FEATURE_COLUMNS; empty DataFrame (with those columns) if
    nothing has been logged yet.
    """
    if not DB_PATH.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    with _connect() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT features_json FROM inference_log ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    records = [json.loads(r[0]) for r in rows]
    return pd.DataFrame(records, columns=FEATURE_COLUMNS)


def count_logged() -> int:
    """Total number of logged inference rows."""
    if not DB_PATH.exists():
        return 0
    with _connect() as conn:
        _ensure_table(conn)
        return int(conn.execute("SELECT COUNT(*) FROM inference_log").fetchone()[0])
