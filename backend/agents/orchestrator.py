"""Orchestrator: a LangGraph state machine over the analysis agents.

Topology (fan-out / fan-in):

                         ┌─> analyst ─┐
                         ├─> forecast ┤
    START ─> fetch ──────┼─> risk ────┼──> judge ─> END
                         └─> sentiment┘

``fetch`` is the single entry point: it downloads the ticker's market data once
and stores it in shared state. From there the four analysis agents fan out and
run in parallel — they are mutually independent (analyst/forecast/risk read the
shared OHLCV; sentiment hits NewsAPI), so there is no reason to serialise them.
``judge`` has an incoming edge from all four, so LangGraph runs it only after
every branch has completed, and it then adjudicates using all four reports.

Parallel nodes write disjoint state keys, so their concurrent state updates
merge without conflict.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.analyst_agent import run_analyst_agent
from backend.agents.forecast_agent import run_forecast_agent
from backend.agents.judge_agent import run_judge_agent
from backend.agents.risk_agent import run_risk_agent
from backend.agents.sentiment_agent import run_sentiment_agent
from backend.data.market_data import MarketData, fetch_market_data
from backend.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalystReport,
    EvidenceSource,
    ForecastReport,
    JudgeVerdict,
    RagRetrieval,
    RetrievalTier,
    RiskReport,
    SentimentReport,
)
from rag.retriever import retrieve as rag_retrieve


class AnalysisState(TypedDict, total=False):
    """Shared state threaded through the analysis graph.

    ``total=False`` because keys are populated progressively as nodes run: the
    entry payload carries only ``ticker``/``lookback_period``; ``fetch`` adds
    ``market_data`` (shared by the analyst/forecast/risk branches); each analysis
    node contributes its own report; ``judge`` adds the verdict.
    """

    ticker: str
    lookback_period: str
    market_data: MarketData
    analyst_report: AnalystReport
    forecast_report: ForecastReport
    risk_report: RiskReport
    sentiment_report: SentimentReport
    rag_retrieval: RagRetrieval
    judge_verdict: JudgeVerdict


# Fixed retrieval query: surfaces the qualitative themes a price model can't see
# (growth, competitive position, enumerated risks) to corroborate the forecast.
RAG_QUERY_TEMPLATE = "{ticker} revenue growth risk factors competitive position"


def fetch_node(state: AnalysisState) -> dict:
    """Single entry point: fetch the ticker's market data once for the fan-out."""
    data = fetch_market_data(state["ticker"], period=state.get("lookback_period", "1y"))
    return {"market_data": data}


def analyst_node(state: AnalysisState) -> dict:
    """Descriptive technical/fundamental read (parallel branch)."""
    return {"analyst_report": run_analyst_agent(state["market_data"])}


def forecast_node(state: AnalysisState) -> dict:
    """Predictive ensemble call (parallel branch)."""
    return {"forecast_report": run_forecast_agent(state["market_data"])}


def risk_node(state: AnalysisState) -> dict:
    """Downside/volatility profile (parallel branch); reuses shared OHLCV, fetches SPY."""
    data = state["market_data"]
    report = run_risk_agent(
        ticker=state["ticker"],
        lookback_period=state.get("lookback_period", "1y"),
        ohlcv=data.ohlcv,
    )
    return {"risk_report": report}


def sentiment_node(state: AnalysisState) -> dict:
    """News-sentiment read via NewsAPI + FinBERT (parallel branch)."""
    return {"sentiment_report": run_sentiment_agent(state["ticker"])}


def rag_node(state: AnalysisState) -> dict:
    """Retrieve 10-K evidence for the ticker (parallel branch).

    Lazily ingests the filing on first use, then maps the retriever's dataclass
    into the API schema. Independent of market_data, so it fans out alongside
    the other agents.
    """
    ticker = state["ticker"]
    result = rag_retrieve(ticker, RAG_QUERY_TEMPLATE.format(ticker=ticker))
    retrieval = RagRetrieval(
        query=result.query,
        evidence_sources=[EvidenceSource(**e) for e in result.evidence],
        retrieval_confidence=result.retrieval_confidence,
        confidence_tier=RetrievalTier(result.confidence_tier),
        note=result.note,
    )
    return {"rag_retrieval": retrieval}


def judge_node(state: AnalysisState) -> dict:
    """Fan-in: adjudicate using all five upstream reports."""
    verdict = run_judge_agent(
        analyst=state["analyst_report"],
        forecast=state["forecast_report"],
        risk=state["risk_report"],
        sentiment=state["sentiment_report"],
        rag=state["rag_retrieval"],
    )
    return {"judge_verdict": verdict}


def _build_graph():
    """Compile the fetch -> {analyst, forecast, risk, sentiment} -> judge graph."""
    graph = StateGraph(AnalysisState)
    graph.add_node("fetch", fetch_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("risk", risk_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("rag", rag_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "fetch")
    # Fan out from the single fetch entry point to the five parallel agents.
    for branch in ("analyst", "forecast", "risk", "sentiment", "rag"):
        graph.add_edge("fetch", branch)
        # Fan in: judge waits for every branch before running.
        graph.add_edge(branch, "judge")
    graph.add_edge("judge", END)

    return graph.compile()


# Compiled once at import; the graph is stateless across invocations.
_GRAPH = _build_graph()


def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """Run the full fan-out/fan-in pipeline for one ticker.

    Invokes the compiled LangGraph, then merges the node outputs into the public
    :class:`AnalysisResponse`, lifting the headline forecast fields to the top
    level and attaching the risk, sentiment and judge results.
    """
    final_state: AnalysisState = _GRAPH.invoke(
        {"ticker": request.ticker, "lookback_period": request.lookback_period}
    )

    analyst = final_state["analyst_report"]
    forecast = final_state["forecast_report"]
    risk = final_state["risk_report"]
    sentiment = final_state["sentiment_report"]
    rag = final_state["rag_retrieval"]
    judge_verdict = final_state["judge_verdict"]

    return AnalysisResponse(
        ticker=analyst.ticker,
        model_version=forecast.model_version,
        prediction=forecast.prediction,
        direction_probability=forecast.direction_probability,
        confidence_interval=forecast.confidence_interval,
        shap_explanation=forecast.shap_explanation,
        analyst=analyst,
        forecast=forecast,
        risk_report=risk,
        sentiment_report=sentiment,
        evidence_sources=rag.evidence_sources,
        retrieval_confidence=rag.retrieval_confidence,
        confidence_tier=rag.confidence_tier,
        judge_verdict=judge_verdict,
    )
