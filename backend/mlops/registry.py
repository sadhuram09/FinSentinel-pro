"""Central MLflow configuration + model-registry helpers.

Two deliberate choices, both forced by how MLflow actually works today:

  * Backend store = local SQLite (``mlflow.db``), not the bare file store. The
    MLflow Model Registry is unavailable on the file store, so a database
    backend is required — SQLite keeps it fully local (no server) while still
    enabling registry features. Run artifacts (plots, models) still land in
    ``mlruns/``.
  * "Staging"/"Production" are expressed as registry **aliases**, not stages.
    MLflow 3 removed the old stage API (``transition_model_version_stage``);
    aliases (``@staging`` / ``@production``) are the supported replacement and
    carry the same promote-gate semantics.

Centralising this here keeps training, promotion and inference in agreement on
the tracking URI, experiment, model name and alias names.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
ARTIFACT_LOCATION = (PROJECT_ROOT / "mlruns").as_uri()

EXPERIMENT_NAME = "finsentinel-forecaster"
MODEL_NAME = "finsentinel-forecaster"
STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"


def configure() -> None:
    """Point MLflow at the local SQLite backend store."""
    mlflow.set_tracking_uri(TRACKING_URI)


def ensure_experiment() -> str:
    """Get-or-create the experiment (artifacts under mlruns/); return its id."""
    configure()
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)


def registry_initialised() -> bool:
    """True if the SQLite registry exists (avoids creating it just to read)."""
    return DB_PATH.exists()
