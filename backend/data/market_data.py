"""yfinance data access, isolated behind a thin function boundary.

Keeping all yfinance calls here means the agents depend on plain pandas/dicts,
not on a third-party client. That makes them trivially testable (inject fixture
data) and lets us swap the data vendor without touching business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class MarketData:
    """Raw market data bundle for one ticker."""

    ticker: str
    ohlcv: pd.DataFrame
    info: dict[str, Any]


def fetch_market_data(ticker: str, period: str = "1y") -> MarketData:
    """Fetch OHLCV history and the fundamentals ``info`` dict for a ticker.

    Args:
        ticker: Equity symbol, e.g. ``"AAPL"``.
        period: yfinance history window (``"6mo"``, ``"1y"``, ``"2y"``, ``"5y"``).

    Returns:
        A :class:`MarketData` bundle.

    Raises:
        ValueError: If yfinance returns no price history (bad/delisted ticker).
    """
    handle = yf.Ticker(ticker)
    ohlcv = handle.history(period=period, auto_adjust=True)
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"No price history returned for ticker '{ticker}'.")

    # ``info`` can be flaky; tolerate failure since fundamentals are optional.
    try:
        info = handle.info or {}
    except Exception:  # noqa: BLE001 - yfinance raises a variety of network errors
        info = {}

    return MarketData(ticker=ticker.upper(), ohlcv=ohlcv, info=info)
