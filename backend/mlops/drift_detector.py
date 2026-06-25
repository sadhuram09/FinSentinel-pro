"""Feature drift detection with Evidently AI.

A model trained on one feature distribution silently decays when live inputs
drift away from it. This compares the distribution of recent inference features
(logged to SQLite by every ``/analysis/run``) against the training baseline
saved during ``scripts/train_models.py``, using Evidently's ``DataDriftPreset``.

It produces a full Evidently HTML report (per-feature drift, distributions) and
returns a compact drift score — the share of features whose distribution drifted
— so the result is consumable both by a human (HTML) and by automation (JSON).

IMPORTANT — small-sample caveat. Drift is a *statistical* comparison of two
distributions. When the current window is tiny (roughly < 30 rows) the
per-feature tests have almost no power: a handful of points cannot represent a
distribution, so they will look "drifted" from the broad training baseline
purely as an artifact of small-sample comparison — NOT because the model has
decayed or the world has shifted. A high drift score on a small window is
therefore not actionable. We surface this via a ``min_sample_warning`` field on
the response so the number is never mistaken for genuine distribution shift; the
detector only becomes trustworthy once enough inferences have accumulated.

Evidently and the report are imported lazily so the (heavy) dependency never
sits on the web app's import path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.ml.models.forecaster import FEATURE_COLUMNS
from backend.mlops.inference_log import load_recent_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "models" / "feature_baseline.csv"
REPORTS_DIR = PROJECT_ROOT / "drift_reports"

# Need at least a couple of live rows for a comparison to even run.
MIN_CURRENT_ROWS = 2
# Below this, drift scores are statistically unreliable (small-sample artifact),
# so we attach a warning rather than letting the number be read as real drift.
RELIABLE_SAMPLE_SIZE = 30
# Convention: flag dataset drift when the majority of features have drifted.
DATASET_DRIFT_SHARE = 0.5


def _load_baseline() -> pd.DataFrame:
    """Load the training feature baseline, or an empty frame if not yet trained."""
    if not BASELINE_PATH.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.read_csv(BASELINE_PATH)[FEATURE_COLUMNS]


def _extract_drift_score(snapshot) -> tuple[float | None, int | None]:
    """Pull (share_of_drifted_features, n_drifted) out of an Evidently snapshot.

    The DataDriftPreset emits a 'drifted columns' metric whose value is a dict
    with ``share`` and ``count``; we locate it structurally so we don't depend on
    Evidently's internal metric ids (which differ across versions).
    """
    payload = snapshot.dict()
    for entry in payload.get("metrics", []):
        value = entry.get("value")
        if isinstance(value, dict) and "share" in value and "count" in value:
            return float(value["share"]), int(value["count"])
    return None, None


def run_drift_report(n: int = 100) -> dict:
    """Compare the last ``n`` inference feature rows to the training baseline.

    Returns a JSON-serialisable dict with the drift score and the path to the
    saved HTML report. Degrades gracefully (``status`` != "ok") when the baseline
    or sufficient live data is missing — it never raises into the endpoint.
    """
    reference = _load_baseline()
    if reference.empty:
        return {
            "status": "no_baseline",
            "note": "No training baseline found; run scripts/train_models.py first.",
        }

    current = load_recent_features(n)
    if len(current) < MIN_CURRENT_ROWS:
        return {
            "status": "insufficient_data",
            "n_current": len(current),
            "note": f"Need at least {MIN_CURRENT_ROWS} logged inferences; call /analysis/run more.",
        }

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = REPORTS_DIR / f"drift_{stamp}.html"
        snapshot.save_html(str(report_path))

        drift_share, n_drifted = _extract_drift_score(snapshot)
    except Exception as exc:  # noqa: BLE001 - surface as status, never 500 the endpoint
        logger.exception("Drift report generation failed.")
        return {"status": "error", "note": f"Evidently drift computation failed: {exc}"}

    result = {
        "status": "ok",
        "drift_score": drift_share,  # share of features whose distribution drifted (0-1)
        "n_drifted_features": n_drifted,
        "n_features": len(FEATURE_COLUMNS),
        "dataset_drift": (drift_share is not None and drift_share >= DATASET_DRIFT_SHARE),
        "n_reference_rows": len(reference),
        "n_current_rows": len(current),
        "report_html": str(report_path),
    }

    # Guard against over-reading a score from too few live samples: below the
    # reliability threshold the comparison is dominated by small-sample noise,
    # so the score reflects sample size, not genuine distribution shift / decay.
    if len(current) < RELIABLE_SAMPLE_SIZE:
        result["min_sample_warning"] = (
            f"Only {len(current)} current rows (< {RELIABLE_SAMPLE_SIZE}). Drift scores from "
            "such small windows are statistically unreliable: a high score here is an artifact "
            "of small-sample comparison, NOT genuine distribution shift or model decay. "
            f"Accumulate at least ~{RELIABLE_SAMPLE_SIZE} inferences before treating drift as real."
        )

    return result
