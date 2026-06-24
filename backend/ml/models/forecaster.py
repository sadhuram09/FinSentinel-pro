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

This module trains lazily on the supplied feature history. In production you
would persist fitted boosters; here we (re)fit on the request's lookback window
so the scaffold runs end-to-end without a model registry.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from backend.schemas import Direction, ForecastReport, HorizonForecast, ShapAttribution
from backend.ml.explainability.shap_explainer import explain_prediction

MODEL_VERSION = "ensemble-v1.0.0"

# Feature columns the ensemble consumes. Kept explicit so SHAP attributions and
# the trained matrix stay aligned.
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

    One pair of boosters is trained per horizon. ``fit`` and ``predict`` operate
    on the engineered technical feature frame produced upstream.
    """

    def __init__(self) -> None:
        self._models: dict[int, tuple[XGBClassifier, LGBMClassifier]] = {}
        self._last_train_features: pd.DataFrame | None = None

    def fit(self, features: pd.DataFrame) -> "StackedForecaster":
        """Fit one XGB+LGBM pair per horizon on the engineered feature frame.

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

            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
            )
            lgbm = LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            xgb.fit(x, y)
            lgbm.fit(x, y)
            self._models[horizon] = (xgb, lgbm)

        self._last_train_features = features
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
            raise RuntimeError("StackedForecaster.predict called before fit().")

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
            model_version=MODEL_VERSION,
            prediction=headline.direction,
            direction_probability=headline.probability_up,
            horizons=horizon_forecasts,
            confidence_interval=(lower, upper),
            shap_explanation=shap_attrs,
        )
