"""Fundamental ratios sourced from yfinance.

Where technicals describe *price behaviour*, fundamentals describe the
*business* behind the price: how it is valued, how profitable it is, and how
much leverage it carries. These slow-moving variables provide regime context
that prevents the model from chasing short-term price noise in isolation.
"""

from __future__ import annotations

from typing import Any

from backend.schemas import FundamentalRatios


def _safe_float(value: Any) -> float | None:
    """Coerce yfinance's loosely-typed ``info`` values to float or None."""
    try:
        if value is None:
            return None
        f = float(value)
        # yfinance occasionally returns NaN/inf for missing fields.
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except (TypeError, ValueError):
        return None


def extract_fundamental_ratios(info: dict[str, Any]) -> FundamentalRatios:
    """Pull valuation, profitability and leverage ratios from a yfinance info dict.

    Args:
        info: The ``yfinance.Ticker(...).info`` mapping.

    Returns:
        A :class:`FundamentalRatios` model. Missing inputs map to ``None`` rather
        than raising, because fundamentals are frequently unavailable for ETFs,
        ADRs and recent IPOs and the pipeline must degrade gracefully.
    """
    # P/E: price paid per dollar of earnings. A high P/E implies the market
    # expects growth; a low one implies value or distress. trailingPE uses
    # realised earnings, which is more robust than forward estimates.
    pe_ratio = _safe_float(info.get("trailingPE"))

    # ROE: net income / shareholder equity. It measures how efficiently the
    # company turns invested capital into profit; durably high ROE signals a
    # competitive moat. yfinance reports it as a fraction (0.25 == 25%).
    roe = _safe_float(info.get("returnOnEquity"))

    # D/E: total debt / equity. It quantifies balance-sheet risk — highly
    # levered firms are more fragile to rate and earnings shocks, which should
    # temper bullish technical signals.
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    # yfinance returns D/E as a percentage (e.g. 150.0); normalise to a ratio.
    if debt_to_equity is not None:
        debt_to_equity = debt_to_equity / 100.0

    return FundamentalRatios(
        pe_ratio=pe_ratio,
        roe=roe,
        debt_to_equity=debt_to_equity,
    )
