"""Analyst agent: produces the *descriptive* view of an asset.

The analyst answers "what is the state of this asset right now?" by combining
technical indicators (recent price behaviour) with fundamental ratios (business
quality and valuation). It deliberately does not predict — that is the forecast
agent's job — so the two concerns stay separable and independently testable.
"""

from __future__ import annotations

from backend.data.market_data import MarketData
from backend.ml.features.fundamental import extract_fundamental_ratios
from backend.ml.features.technical import latest_technical_snapshot
from backend.schemas import AnalystReport, FundamentalRatios, TechnicalIndicators


def _build_commentary(
    ticker: str, tech: TechnicalIndicators, fund: FundamentalRatios
) -> str:
    """Compose a short, rule-based reading of the indicators.

    This is intentionally deterministic (no LLM) so the scaffold is reproducible;
    swap in a generative model later without changing the agent's contract.
    """
    notes: list[str] = []

    if tech.rsi >= 70:
        notes.append(f"RSI {tech.rsi:.0f} is overbought (possible pullback).")
    elif tech.rsi <= 30:
        notes.append(f"RSI {tech.rsi:.0f} is oversold (possible bounce).")
    else:
        notes.append(f"RSI {tech.rsi:.0f} is neutral.")

    if tech.macd_hist > 0:
        notes.append("MACD histogram positive: upward momentum.")
    else:
        notes.append("MACD histogram negative: downward momentum.")

    if tech.bb_pct >= 1.0:
        notes.append("Price above the upper Bollinger Band (stretched).")
    elif tech.bb_pct <= 0.0:
        notes.append("Price below the lower Bollinger Band (stretched).")

    if fund.pe_ratio is not None:
        notes.append(f"Trailing P/E {fund.pe_ratio:.1f}.")
    if fund.roe is not None:
        notes.append(f"ROE {fund.roe * 100:.1f}%.")
    if fund.debt_to_equity is not None:
        notes.append(f"D/E {fund.debt_to_equity:.2f}.")

    return f"{ticker}: " + " ".join(notes)


def run_analyst_agent(data: MarketData) -> AnalystReport:
    """Run the analyst over a market-data bundle and return an AnalystReport."""
    tech = latest_technical_snapshot(data.ohlcv)
    fund = extract_fundamental_ratios(data.info)

    last_bar = data.ohlcv.dropna(subset=["Close"]).iloc[-1]
    as_of = data.ohlcv.index[-1].date().isoformat()

    return AnalystReport(
        ticker=data.ticker,
        as_of=as_of,
        last_close=float(last_bar["Close"]),
        technicals=tech,
        fundamentals=fund,
        commentary=_build_commentary(data.ticker, tech, fund),
    )
