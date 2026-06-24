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
    judge_verdict: JudgeVerdict


class AnalysisRequest(BaseModel):
    """Request body for POST /analysis/run."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Equity ticker, e.g. 'AAPL'.")
    lookback_period: Literal["6mo", "1y", "2y", "5y"] = Field(
        "1y", description="History window pulled from yfinance for feature computation."
    )
