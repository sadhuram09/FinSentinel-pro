"""Promote the Staging ensemble to Production in the MLflow registry.

Training registers each new ensemble under the ``@staging`` alias. Promotion is
a deliberate, gated step: a human reviews the staged version's metrics and only
then moves the ``@production`` alias to it. The inference path
(``forecaster.py``) loads ``@production``, so nothing reaches live traffic until
this runs.

(MLflow 3 removed model *stages*; ``@staging`` / ``@production`` aliases are the
supported equivalent and carry the same promote-gate meaning.)

    python scripts/promote_model.py          # interactive confirmation
    python scripts/promote_model.py --yes     # non-interactive (CI/automation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402

from backend.mlops import registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote Staging model to Production.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    args = parser.parse_args()

    registry.configure()
    client = mlflow.MlflowClient()

    try:
        staged = client.get_model_version_by_alias(registry.MODEL_NAME, registry.STAGING_ALIAS)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"No '@{registry.STAGING_ALIAS}' model found for '{registry.MODEL_NAME}': {exc}")

    run = client.get_run(staged.run_id) if staged.run_id else None
    print(f"Model     : {registry.MODEL_NAME}")
    print(f"Staging   : version {staged.version} (run {staged.run_id})")
    if run is not None:
        for key in ("auc_ensemble_h5", "auc_ensemble_h30", "brier_ensemble_h5", "brier_ensemble_h30"):
            if key in run.data.metrics:
                print(f"    {key}: {run.data.metrics[key]:.4f}")

    # Show what is currently in production, if anything.
    try:
        current = client.get_model_version_by_alias(registry.MODEL_NAME, registry.PRODUCTION_ALIAS)
        print(f"Current production: version {current.version}")
    except Exception:  # noqa: BLE001 - no production yet
        print("Current production: none")

    if not args.yes:
        reply = input(f"\nPromote version {staged.version} to '@{registry.PRODUCTION_ALIAS}'? [y/N] ").strip().lower()
        if reply != "y":
            sys.exit("Aborted; production unchanged.")

    client.set_registered_model_alias(registry.MODEL_NAME, registry.PRODUCTION_ALIAS, staged.version)
    print(f"\nPromoted version {staged.version} -> '@{registry.PRODUCTION_ALIAS}'.")
    print("Restart the API to pick up the new production model.")


if __name__ == "__main__":
    main()
