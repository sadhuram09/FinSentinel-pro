"""Shared Pydantic v2 models for agent and API outputs.

Centralising these contracts keeps the agents, the orchestrator and the
HTTP layer in agreement about field names and types, and gives us one place
to evolve the public API surface.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Discrete directional call for a forecast horizon.

    We collapse a continuous price forecast into three buckets because
    downstream consumers (alerts, dashboards) act on direction, not on a
    precise price target that would imply false precision.
    """

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class TechnicalIndicators(BaseModel):
    """Snapshot of momentum / trend / volatility indicators.

    Technical indicators summarise recent price action so a model does not
    have to relearn common market patterns from raw OHLCV data.
    """

    rsi: float = Field(..., description="Relative Strength Index (0-100); >70 overbought, <30 oversold.")
    macd: float = Field(..., description="MACD line: fast EMA minus slow EMA; trend momentum.")
    macd_signal: float = Field(..., description="Signal line (EMA of MACD); crossovers flag momentum shifts.")
    macd_hist: float = Field(..., description="MACD histogram: MACD minus signal; momentum acceleration.")
    bb_upper: float = Field(..., description="Upper Bollinger Band; price near it suggests stretched upside.")
    bb_middle: float = Field(..., description="Middle Bollinger Band (SMA); the mean price reverts toward.")
    bb_lower: float = Field(..., description="Lower Bollinger Band; price near it suggests stretched downside.")
    bb_pct: float = Field(..., description="Position of price within the bands (0=lower, 1=upper).")


class FundamentalRatios(BaseModel):
    """Valuation / profitability / leverage ratios.

    Fundamentals provide the slow-moving context technicals lack: whether a
    company is cheap, profitable and solvent, which conditions how much weight
    to put on short-term price signals.
    """

    pe_ratio: float | None = Field(None, description="Price/Earnings; how much investors pay per unit of earnings.")
    roe: float | None = Field(None, description="Return on Equity; profitability relative to shareholder capital.")
    debt_to_equity: float | None = Field(None, description="Debt/Equity; leverage and balance-sheet risk.")


class AnalystReport(BaseModel):
    """Output of the analyst agent: the descriptive state of the asset."""

    ticker: str
    as_of: str = Field(..., description="ISO date of the latest bar used.")
    last_close: float
    technicals: TechnicalIndicators
    fundamentals: FundamentalRatios
    commentary: str = Field(..., description="Human-readable synthesis of the signals.")


class ShapAttribution(BaseModel):
    """A single feature's contribution to a prediction."""

    feature: str
    value: float = Field(..., description="The feature's input value for this sample.")
    contribution: float = Field(..., description="Signed SHAP value; push toward (+) or away (-) from 'up'.")


class HorizonForecast(BaseModel):
    """Directional forecast for one time horizon."""

    horizon_days: int
    direction: Direction
    probability_up: float = Field(..., ge=0.0, le=1.0, description="Model probability that direction is 'up'.")


class ForecastReport(BaseModel):
    """Output of the forecast agent: the predictive view."""

    ticker: str
    model_version: str
    prediction: Direction = Field(..., description="Headline (5-day) directional call.")
    direction_probability: float = Field(..., ge=0.0, le=1.0)
    horizons: list[HorizonForecast]
    confidence_interval: tuple[float, float] = Field(
        ..., description="Lower/upper probability bound reflecting ensemble disagreement."
    )
    shap_explanation: list[ShapAttribution]


class Verdict(str, Enum):
    """Governance decision the judge agent renders over a forecast.

    The judge is a supervisory layer: it does not predict, it adjudicates
    whether the analyst's descriptive read and the forecaster's predictive call
    are coherent enough to act on. Three states map cleanly onto downstream
    routing — trade, escalate-to-human, or discard.
    """

    APPROVED = "Approved"
    CONFLICTED = "Conflicted"
    REJECTED = "Rejected"


class JudgeVerdict(BaseModel):
    """Output of the judge agent: an auditable adjudication of the forecast.

    Every verdict carries the quantitative basis (``consistency_score``), the
    qualitative red flags, a plain-English justification, and an ``audit_id`` so
    each decision is independently traceable in a compliance log.
    """

    verdict: Verdict
    consistency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Agreement between analyst and forecast direction (1=aligned)."
    )
    flags: list[str] = Field(
        default_factory=list, description="Qualitative risk flags, e.g. 'wide uncertainty', 'low confidence'."
    )
    override_reasoning: str = Field(..., description="LLM-generated justification of the verdict.")
    audit_id: str = Field(..., description="UUID4 uniquely identifying this adjudication for audit trails.")


class RiskTier(str, Enum):
    """Coarse risk bucket used by the judge to weight a forecast.

    A directional call made on a fragile, high-volatility name deserves more
    scrutiny than the same call on a stable one. Collapsing the risk metrics
    into three tiers gives the judge a single lever without it having to
    re-derive thresholds from raw statistics.
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RiskReport(BaseModel):
    """Output of the risk agent: the asset's downside/volatility profile.

    These metrics matter to the judge because a forecast's *consequence* depends
    on the risk regime: the same probability of a down move is far more
    dangerous in a name with fat left-tail losses and deep drawdowns.
    """

    ticker: str
    lookback_period: str
    var_95: float = Field(
        ..., description="1-day historical VaR at 95% as a positive loss fraction (e.g. 0.03 = 3%)."
    )
    var_99: float = Field(..., description="1-day historical VaR at 99% (deeper tail) as a positive loss fraction.")
    cvar_95: float = Field(
        ..., description="Expected Shortfall at 95%: average loss *given* the worst 5% of days (>= VaR95)."
    )
    sharpe_ratio: float = Field(..., description="Annualised excess return per unit of total volatility.")
    sortino_ratio: float = Field(..., description="Annualised excess return per unit of downside volatility only.")
    max_drawdown: float = Field(..., description="Largest peak-to-trough decline over the window, positive fraction.")
    beta: float = Field(..., description="Sensitivity to SPY: >1 amplifies market moves, <1 dampens them.")
    risk_tier: RiskTier


class SentimentLabel(str, Enum):
    """Discrete sentiment polarity for a headline or an aggregate."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class HeadlineSentiment(BaseModel):
    """One headline scored by FinBERT."""

    headline: str
    label: SentimentLabel
    score: float = Field(..., ge=-1.0, le=1.0, description="Signed confidence: +pos, -neg, 0 neutral.")


class SentimentReport(BaseModel):
    """Output of the sentiment agent: the recent news narrative around the asset.

    News sentiment is an orthogonal signal to price/technicals: it can corroborate
    a forecast (bullish forecast + bullish coverage) or flag a divergence the
    judge should weigh (bullish forecast against a wave of negative headlines).
    """

    ticker: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Aggregate polarity, -1 (bearish) to +1 (bullish).")
    sentiment_label: SentimentLabel
    headline_count: int = Field(..., description="Number of headlines analysed in the window.")
    top_headlines: list[HeadlineSentiment] = Field(
        default_factory=list, description="Up to 3 highest-magnitude (most impactful) headlines."
    )
    note: str | None = Field(None, description="Set when sentiment is degraded (e.g. no API key / no news).")


class AnalysisResponse(BaseModel):
    """Combined response returned by POST /analysis/run."""

    ticker: str
    model_version: str
    prediction: Direction
    direction_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    shap_explanation: list[ShapAttribution]
    analyst: AnalystReport
    forecast: ForecastReport
    risk_report: RiskReport
    sentiment_report: SentimentReport
    judge_verdict: JudgeVerdict


class AnalysisRequest(BaseModel):
    """Request body for POST /analysis/run."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Equity ticker, e.g. 'AAPL'.")
    lookback_period: Literal["6mo", "1y", "2y", "5y"] = Field(
        "1y", description="History window pulled from yfinance for feature computation."
    )
