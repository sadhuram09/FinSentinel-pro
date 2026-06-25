"""Forecast agent: produces the *predictive* view of an asset.

It engineers the technical feature matrix and delegates to the forecaster, which
serves a pre-trained, persisted ensemble when one is available (and otherwise
falls back to fitting on the request window). It emits directional probabilities
for the 5-day and 30-day horizons together with a SHAP explanation. Separating
this from the analyst keeps the "what will happen" reasoning isolated from
"what is happening".
"""

from __future__ import annotations

from backend.data.market_data import MarketData
from backend.ml.features.technical import compute_technical_features
from backend.ml.models.forecaster import forecast
from backend.schemas import ForecastReport


def run_forecast_agent(data: MarketData) -> ForecastReport:
    """Engineer features and produce a ForecastReport from the forecaster.

    The forecaster loads a persisted, versioned model trained offline by
    ``scripts/train_models.py``; only when no such model exists does it fit on
    the request's lookback window (logged as ``-fit-per-request``).
    """
    features = compute_technical_features(data.ohlcv)
    return forecast(ticker=data.ticker, features=features)
