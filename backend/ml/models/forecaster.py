"""Stacked ensemble forecaster: XGBoost (primary) + LightGBM (secondary).

Why an ensemble of two gradient-boosted tree libraries?
  * XGBoost and LightGBM make different bias/variance trade-offs (level-wise
    vs leaf-wise growth, different regularisation defaults). Averaging their
    probabilities reduces model-specific variance and tends to calibrate
    better than either alone.
  * Both are strong on the small, tabular, noisy datasets typical of financial
    features, and both expose the tree structure SHAP needs for fast exact
    attributions.

The model frames direction as binary classification (P(up)) per horizon, and
derives a 3-way Direction by applying a neutral band around 0.5 so we don't
emit confident calls on coin-flip probabilities.

Production flow: boosters are trained offline by ``scripts/train_models.py`` on
a pooled, multi-year, multi-ticker dataset and persisted to ``models/*.joblib``.
This module loads those at import time. If no persisted model is found (e.g. a
fresh clone before training has run), it falls back to fitting per request on
the caller's lookback window so the scaffold still works end-to-end — but logs a
clear warning, because per-request fitting overfits a single ticker's recent
history and is not how the system is meant to run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from backend.schemas import Direction, ForecastReport, HorizonForecast, ShapAttribution
from backend.ml.explainability.shap_explainer import explain_prediction

logger = logging.getLogger(__name__)

MODEL_VERSION = "ensemble-v1.0.0"

# Project-root ``models/`` directory holding persisted boosters.
# forecaster.py lives at backend/ml/models/, so the project root is parents[3].
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

# Feature columns the ensemble consumes. Kept explicit so SHAP attributions and
# the trained matrix stay aligned.
#
# NOTE: intentionally technical-only. Fundamental ratios (P/E, ROE, D/E) are NOT
# in the predictive matrix because yfinance exposes only the *current* snapshot,
# not a point-in-time history — using today's P/E as a feature for a label from
# three years ago would leak future information into training. Fundamentals are
# instead consumed by the analyst and judge layers. Training and inference must
# use exactly this set, or a persisted model will not load against live data.
FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_pct",
]

# Ensemble weights. XGBoost is the primary learner, so it carries more weight;
# LightGBM diversifies. Weights sum to 1 so the output stays a probability.
XGB_WEIGHT = 0.6
LGBM_WEIGHT = 0.4

# Probabilities inside [0.5 - BAND, 0.5 + BAND] are treated as "flat": the model
# is not confident enough to commit to a direction.
NEUTRAL_BAND = 0.05

HORIZONS = (5, 30)


def model_path(horizon: int, models_dir: Path = MODELS_DIR) -> Path:
    """Canonical path of the persisted, ticker-agnostic bundle for a horizon."""
    return models_dir / f"ensemble_h{horizon}.joblib"


def build_xgb() -> XGBClassifier:
    """Construct an unfitted XGBoost classifier with the project's hyperparameters.

    Centralised so offline training (scripts/train_models.py) and the
    fit-per-request fallback produce identical model configurations.
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )


def build_lgbm() -> LGBMClassifier:
    """Construct an unfitted LightGBM classifier with the project's hyperparameters."""
    return LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def _build_labels(close: pd.Series, horizon: int) -> pd.Series:
    """Binary label: 1 if close rises over the next ``horizon`` trading days."""
    future = close.shift(-horizon)
    return (future > close).astype(int)


def _direction_from_prob(prob_up: float) -> Direction:
    """Map a P(up) into the 3-way Direction using the neutral band."""
    if prob_up >= 0.5 + NEUTRAL_BAND:
        return Direction.UP
    if prob_up <= 0.5 - NEUTRAL_BAND:
        return Direction.DOWN
    return Direction.FLAT


class StackedForecaster:
    """Weighted-average ensemble of XGBoost and LightGBM classifiers.

    One pair of boosters is held per horizon. The pair is either loaded from a
    persisted bundle (production path) or fitted in-memory (fallback path);
    ``predict`` behaves identically either way.
    """

    def __init__(self, model_version: str = MODEL_VERSION) -> None:
        self._models: dict[int, tuple[XGBClassifier, LGBMClassifier]] = {}
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    @classmethod
    def load_pretrained(
        cls, models_dir: Path = MODELS_DIR
    ) -> "StackedForecaster | None":
        """Load persisted boosters for every horizon, or return None if incomplete.

        A bundle is a dict with ``xgb``/``lgbm`` estimators plus metadata
        (``feature_columns``, ``model_version``). If any horizon's file is
        missing the whole load is treated as unavailable, so we never serve a
        partially-loaded ensemble.
        """
        if not models_dir.exists():
            return None

        models: dict[int, tuple[XGBClassifier, LGBMClassifier]] = {}
        version = MODEL_VERSION
        for horizon in HORIZONS:
            path = model_path(horizon, models_dir)
            if not path.exists():
                logger.warning("Persisted model missing for horizon %sd: %s", horizon, path)
                return None
            bundle = joblib.load(path)
            cols = bundle.get("feature_columns")
            if cols is not None and list(cols) != FEATURE_COLUMNS:
                # A feature-schema drift would silently corrupt predictions.
                logger.error(
                    "Persisted model %s has feature columns %s, expected %s; ignoring.",
                    path, cols, FEATURE_COLUMNS,
                )
                return None
            models[horizon] = (bundle["xgb"], bundle["lgbm"])
            version = bundle.get("model_version", MODEL_VERSION)

        instance = cls(model_version=version)
        instance._models = models
        return instance

    def fit(self, features: pd.DataFrame) -> "StackedForecaster":
        """Fit one XGB+LGBM pair per horizon on the engineered feature frame.

        Used only by the fallback path (and tests). The model version is tagged
        ``-fit-per-request`` so the API response makes the provenance obvious.

        Args:
            features: Frame containing :data:`FEATURE_COLUMNS` plus a ``Close``
                column, indexed by date.
        """
        for horizon in HORIZONS:
            labels = _build_labels(features["Close"], horizon)
            # Drop the tail rows whose future label is unknown and any warm-up
            # rows with NaN indicators.
            frame = features[FEATURE_COLUMNS].join(labels.rename("y")).dropna()
            x = frame[FEATURE_COLUMNS]
            y = frame["y"].to_numpy()

            xgb = build_xgb()
            lgbm = build_lgbm()
            xgb.fit(x, y)
            lgbm.fit(x, y)
            self._models[horizon] = (xgb, lgbm)

        self._model_version = f"{MODEL_VERSION}-fit-per-request"
        return self

    def _predict_proba(self, horizon: int, x: pd.DataFrame) -> float:
        """Weighted-average P(up) from both boosters for one sample."""
        xgb, lgbm = self._models[horizon]
        p_xgb = float(xgb.predict_proba(x)[0, 1])
        p_lgbm = float(lgbm.predict_proba(x)[0, 1])
        return XGB_WEIGHT * p_xgb + LGBM_WEIGHT * p_lgbm

    def predict(self, ticker: str, features: pd.DataFrame) -> ForecastReport:
        """Produce a multi-horizon directional forecast for the latest bar.

        The 5-day horizon is treated as the headline prediction; the 30-day
        horizon adds a longer-term view. The confidence interval is derived
        from the disagreement between the two boosters on the headline horizon
        — wider spread means less certainty.
        """
        if not self._models:
            raise RuntimeError("StackedForecaster.predict called before fit/load.")

        latest = features[FEATURE_COLUMNS].dropna().iloc[[-1]]
        x = latest

        horizon_forecasts: list[HorizonForecast] = []
        for horizon in HORIZONS:
            prob_up = self._predict_proba(horizon, x)
            horizon_forecasts.append(
                HorizonForecast(
                    horizon_days=horizon,
                    direction=_direction_from_prob(prob_up),
                    probability_up=prob_up,
                )
            )

        headline = next(h for h in horizon_forecasts if h.horizon_days == 5)

        # Confidence interval from inter-model spread on the headline horizon.
        xgb, lgbm = self._models[5]
        p_xgb = float(xgb.predict_proba(x)[0, 1])
        p_lgbm = float(lgbm.predict_proba(x)[0, 1])
        lower = max(0.0, min(p_xgb, p_lgbm))
        upper = min(1.0, max(p_xgb, p_lgbm))

        # SHAP attributions explain the headline (5-day) XGBoost model, the
        # primary learner, so the explanation matches the dominant signal.
        shap_attrs: list[ShapAttribution] = explain_prediction(
            model=xgb,
            sample=latest,
            feature_columns=FEATURE_COLUMNS,
        )

        return ForecastReport(
            ticker=ticker,
            model_version=self._model_version,
            prediction=headline.direction,
            direction_probability=headline.probability_up,
            horizons=horizon_forecasts,
            confidence_interval=(lower, upper),
            shap_explanation=shap_attrs,
        )


class EnsembleModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper so the whole multi-horizon ensemble is one registry model.

    Holding both horizons' boosters in a single registered model means one
    Production alias governs the entire forecaster. ``predict`` returns per-
    horizon P(up); the forecaster unwraps this back to the raw boosters so SHAP
    still operates on the underlying trees.
    """

    def __init__(
        self,
        models: dict[int, tuple] | None = None,
        feature_columns: list[str] | None = None,
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.models = models or {}
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.model_version = model_version

    def predict(self, context, model_input, params=None):  # noqa: D102 - pyfunc contract
        x = model_input[self.feature_columns]
        out = {}
        for horizon, (xgb, lgbm) in self.models.items():
            out[f"prob_up_h{horizon}"] = (
                XGB_WEIGHT * xgb.predict_proba(x)[:, 1] + LGBM_WEIGHT * lgbm.predict_proba(x)[:, 1]
            )
        return pd.DataFrame(out, index=model_input.index)


def _load_from_registry() -> "StackedForecaster | None":
    """Load the ``@production`` ensemble from the MLflow registry, or None."""
    from backend.mlops import registry

    if not registry.registry_initialised():
        return None
    try:
        registry.configure()
        uri = f"models:/{registry.MODEL_NAME}@{registry.PRODUCTION_ALIAS}"
        impl = mlflow.pyfunc.load_model(uri).unwrap_python_model()
        forecaster = StackedForecaster(model_version=impl.model_version)
        forecaster._models = impl.models
        return forecaster
    except Exception:  # noqa: BLE001 - no Production alias yet, or load failure
        logger.info("No @production model in registry (or load failed); using fallback.")
        return None


def _initialise_forecaster() -> "StackedForecaster | None":
    """Resolve the active forecaster: registry Production -> local joblib -> none."""
    from_registry = _load_from_registry()
    if from_registry is not None:
        logger.info("Loaded forecaster '%s' from MLflow registry (@production).", from_registry.model_version)
        return from_registry

    from_disk = StackedForecaster.load_pretrained()
    if from_disk is not None:
        logger.info("Loaded pretrained forecaster '%s' from %s.", from_disk.model_version, MODELS_DIR)
        return from_disk

    logger.warning(
        "No registry @production model and no persisted models in %s. Falling back "
        "to per-request fitting (scaffold mode); run `python scripts/train_models.py`.",
        MODELS_DIR,
    )
    return None


# Resolve the active model once at import.
_PRETRAINED: StackedForecaster | None = _initialise_forecaster()


def forecast(ticker: str, features: pd.DataFrame) -> ForecastReport:
    """Predict using the persisted model, or fit-per-request if none is loaded.

    This is the single entry point the forecast agent should call. It hides the
    pretrained-vs-fallback decision so callers never accidentally retrain when a
    persisted model is available.
    """
    if _PRETRAINED is not None:
        return _PRETRAINED.predict(ticker=ticker, features=features)

    logger.warning(
        "Serving '%s' for %s by fitting on the request window — no persisted "
        "model available.", ticker, f"{MODEL_VERSION}-fit-per-request",
    )
    return StackedForecaster().fit(features).predict(ticker=ticker, features=features)
