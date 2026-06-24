"""Forecast agent: produces the *predictive* view of an asset.

It engineers the technical feature matrix, fits the stacked ensemble on the
supplied history, and emits directional probabilities for the 5-day and 30-day
horizons together with a SHAP explanation. Separating this from the analyst
keeps the "what will happen" reasoning isolated from "what is happening".
"""

from __future__ import annotations

from backend.data.market_data import MarketData
from backend.ml.features.technical import compute_technical_features
from backend.ml.models.forecaster import StackedForecaster
from backend.schemas import ForecastReport


def run_forecast_agent(data: MarketData) -> ForecastReport:
    """Engineer features, fit the ensemble, and return a ForecastReport.

    The model is fit on the request's lookback window. For a scaffold this keeps
    everything self-contained; in production you would load a pre-trained,
    versioned model from a registry instead of refitting per request.
    """
    features = compute_technical_features(data.ohlcv)
    forecaster = StackedForecaster().fit(features)
    return forecaster.predict(ticker=data.ticker, features=features)
