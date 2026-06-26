"""Cold-start model bootstrap for the FastAPI app.

On a fresh deployment (e.g. a Render free-tier instance whose disk is wiped on
every redeploy) there is no persisted model and no MLflow registry, so the
forecaster would silently fall back to overfit, per-request fitting. To avoid
that, :func:`ensure_production_model` runs at startup: if no ``@production``
model is registered it trains one inline — using the *same* pipeline as
``scripts/train_models.py`` — registers it, and promotes it to ``@production``
before the app accepts traffic.

The training step is slow (it downloads several years of OHLCV and fits four
boosters), which on a cold start looks exactly like a hang. We therefore log
loud, unambiguous progress messages around it so the wait is explained rather
than mysterious.

When a ``@production`` model already exists (local dev, or a redeploy within a
persistent session) this is a couple of cheap registry reads and the app boots
straight through.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from backend.mlops import registry

logger = logging.getLogger(__name__)

# scripts/ is a sibling of backend/ at the project root and is not an installed
# package, so make the project root importable before pulling in the trainer.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_production_model() -> bool:
    """Guarantee a ``@production`` model exists, training one on startup if not.

    Returns True if a fresh model was trained and promoted during this call, and
    False if an existing ``@production`` model was found and reused. Any failure
    is logged and swallowed (returns False): a training error must not take the
    whole API down, since the forecaster still has its per-request fallback.
    """
    if registry.production_model_exists():
        logger.info(
            "Found an existing @production model in the registry - skipping "
            "startup training and booting normally."
        )
        return False

    logger.warning(
        "No @production model found - training on startup, this may take a "
        "minute... (downloading price history and fitting the ensemble; the "
        "server will not accept requests until this finishes)."
    )
    started = time.monotonic()
    try:
        # Imported lazily: this pulls in the full ML training stack, which we do
        # not want to load when an existing model lets us skip training.
        from scripts.train_models import run_training

        new_version = run_training()
        logger.info(
            "Startup training complete (%.1fs) - registered version %s as "
            "@staging; promoting to @production...",
            time.monotonic() - started,
            new_version,
        )
        registry.promote_to_production(new_version)
        logger.info(
            "Promoted version %s to @production. The API is ready to serve.",
            new_version,
        )
        return True
    except Exception:  # noqa: BLE001 - never let a training failure block boot
        logger.exception(
            "Startup training failed after %.1fs. Booting anyway; the forecaster "
            "will use its per-request fallback until a model is trained.",
            time.monotonic() - started,
        )
        return False
