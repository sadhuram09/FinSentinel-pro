"""MLOps router: on-demand drift monitoring.

Thin HTTP layer over the drift detector — validate nothing, delegate, return the
score. The check is a manual trigger (not a background job) so an operator can
run it whenever they want a fresh read on feature drift.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.mlops.drift_detector import run_drift_report
from backend.mlops.inference_log import count_logged

router = APIRouter(prefix="/mlops", tags=["mlops"])


@router.get("/drift-report", summary="Run feature-drift check against the training baseline")
def drift_report(n: int = 100) -> dict:
    """Compare the last ``n`` logged inference feature rows to the training baseline.

    Returns the drift score (share of drifted features), dataset-drift flag, row
    counts, and the path to the generated Evidently HTML report. When fewer than
    ~30 inference rows have been logged, the response also carries a
    ``min_sample_warning``: drift on such a small window is a small-sample
    artifact, not genuine distribution shift, and should not be acted on.
    """
    result = run_drift_report(n=n)
    result["total_logged_inferences"] = count_logged()
    return result
