"""Risk agent: quantifies the asset's downside and volatility profile.

The forecast agent says *which way* and *how likely*; the risk agent says *how
much it could hurt if wrong*. The judge needs both: a 55%-up call on a placid
blue-chip and the same call on a name that routinely drops 8% in a day are not
equally actionable. Every metric here exists to give the judge a different facet
of that downside picture.

All metrics are computed from daily, split/dividend-adjusted closes over the
requested lookback window. Beta additionally pulls SPY over the same window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from backend.schemas import RiskReport, RiskTier

# Annual risk-free rate proxy (~short-term Treasury). Held constant for now;
# a later phase can source the live 3-month bill from FRED.
RISK_FREE_ANNUAL = 0.045
TRADING_DAYS = 252
RISK_FREE_DAILY = RISK_FREE_ANNUAL / TRADING_DAYS

# risk_tier thresholds. Tuned for single-name US equities: a >4% one-day VaR or a
# >40% drawdown marks a genuinely volatile name; <2% VaR and <20% drawdown is
# placid. Anything between is Medium. Kept deliberately simple and explicit so
# the judge's behaviour is auditable.
HIGH_VAR95 = 0.04
HIGH_MAX_DD = 0.40
LOW_VAR95 = 0.02
LOW_MAX_DD = 0.20


def _daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns, NaNs dropped."""
    return close.pct_change().dropna()


def _historical_var(returns: pd.Series, confidence: float) -> float:
    """Historical-method VaR as a positive loss fraction.

    VaR answers "on a normal-to-bad day, how much could I lose?". The historical
    method makes no distributional assumption — it just reads the empirical loss
    quantile, so it captures the fat tails that sink Gaussian VaR. Returned as a
    positive number (a 3% VaR is reported as 0.03).
    """
    percentile = (1.0 - confidence) * 100.0
    return float(-np.percentile(returns, percentile))


def _cvar(returns: pd.Series, confidence: float) -> float:
    """Conditional VaR / Expected Shortfall as a positive loss fraction.

    CVaR matters where VaR is blind: VaR is the *threshold* of the bad tail,
    CVaR is the *average loss once you are in it*. Two assets can share a VaR but
    have very different CVaRs; the one with the deeper average tail loss is the
    more dangerous, which is exactly what the judge should penalise.
    """
    percentile = (1.0 - confidence) * 100.0
    threshold = np.percentile(returns, percentile)
    tail = returns[returns <= threshold]
    if tail.empty:
        return float(-threshold)
    return float(-tail.mean())


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio: excess return per unit of *total* volatility.

    A high directional conviction is worth less if the asset's risk-adjusted
    return history is poor, so the judge can read Sharpe as the quality of the
    risk the forecast is asking it to take.
    """
    excess = returns - RISK_FREE_DAILY
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    """Annualised Sortino ratio: excess return per unit of *downside* volatility.

    Sharpe punishes upside and downside swings alike; Sortino only counts harmful
    (below-target) volatility. Comparing the two tells the judge whether an
    asset's risk is mostly benign upside churn or genuine downside.
    """
    excess = returns - RISK_FREE_DAILY
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside**2).mean()) if not downside.empty else 0.0
    if downside_dev == 0 or np.isnan(downside_dev):
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS))


def _max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline over the window, as a positive fraction.

    Drawdown is the metric an actual holder feels: it is the worst loss endured
    from a prior high. A deep historical drawdown warns the judge how violent
    this name's adverse moves can get, independent of day-to-day volatility.
    """
    cumulative = (1.0 + _daily_returns(close)).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(-drawdown.min()) if not drawdown.empty else 0.0


def _beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta vs SPY via covariance/variance on date-aligned returns.

    Beta tells the judge how much of the asset's risk is undiversifiable market
    risk: a beta of 1.5 means the name tends to move 1.5x the market, so a
    bullish call is implicitly a leveraged bet on the market itself.
    """
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    stock, market = aligned.iloc[:, 0], aligned.iloc[:, 1]
    market_var = market.var()
    if market_var == 0 or np.isnan(market_var):
        return float("nan")
    return float(stock.cov(market) / market_var)


def _classify_tier(var_95: float, max_dd: float) -> RiskTier:
    """Map VaR95 + max drawdown onto a coarse risk tier (see module thresholds)."""
    if var_95 >= HIGH_VAR95 or max_dd >= HIGH_MAX_DD:
        return RiskTier.HIGH
    if var_95 <= LOW_VAR95 and max_dd <= LOW_MAX_DD:
        return RiskTier.LOW
    return RiskTier.MEDIUM


def run_risk_agent(
    ticker: str,
    lookback_period: str = "1y",
    ohlcv: pd.DataFrame | None = None,
) -> RiskReport:
    """Compute the full risk profile for a ticker over the lookback window.

    Args:
        ticker: Equity symbol.
        lookback_period: yfinance period string used for both the ticker (if not
            supplied) and the SPY benchmark.
        ohlcv: Optional pre-fetched OHLCV for the ticker (reused from the graph's
            shared fetch to avoid a redundant download). SPY is always fetched
            here since it is benchmark data the other agents do not need.

    Returns:
        A :class:`RiskReport`.

    Raises:
        ValueError: If no usable price history is available for the ticker.
    """
    if ohlcv is None or ohlcv.empty:
        ohlcv = yf.Ticker(ticker).history(period=lookback_period, auto_adjust=True)
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"No price history for risk analysis of '{ticker}'.")

    close = ohlcv["Close"].dropna()
    returns = _daily_returns(close)

    spy = yf.Ticker("SPY").history(period=lookback_period, auto_adjust=True)
    spy_returns = _daily_returns(spy["Close"]) if spy is not None and not spy.empty else pd.Series(dtype=float)

    var_95 = _historical_var(returns, 0.95)
    max_dd = _max_drawdown(close)

    return RiskReport(
        ticker=ticker.upper(),
        lookback_period=lookback_period,
        var_95=var_95,
        var_99=_historical_var(returns, 0.99),
        cvar_95=_cvar(returns, 0.95),
        sharpe_ratio=_sharpe(returns),
        sortino_ratio=_sortino(returns),
        max_drawdown=max_dd,
        beta=_beta(returns, spy_returns),
        risk_tier=_classify_tier(var_95, max_dd),
    )
