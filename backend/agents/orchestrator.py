"""Orchestrator: a LangGraph state machine over the three agents.

The pipeline is a linear graph — analyst -> forecast -> judge -> END — sharing
one typed state object. Expressing it as a ``StateGraph`` (rather than plain
function calls) makes the data flow explicit and auditable, and leaves room to
add conditional edges later (e.g. short-circuit to END on a data-fetch failure,
or loop back for a second forecast when the judge is Conflicted).

Market data is fetched once in the analyst node and threaded through the state
so the forecast node reuses it rather than hitting yfinance twice.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from backend.agents.analyst_agent import run_analyst_agent
from backend.agents.forecast_agent import run_forecast_agent
from backend.agents.judge_agent import run_judge_agent
from backend.data.market_data import MarketData, fetch_market_data
from backend.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalystReport,
    ForecastReport,
    JudgeVerdict,
)


class AnalysisState(TypedDict, total=False):
    """Shared state threaded through the analysis graph.

    ``total=False`` because keys are populated progressively as nodes run:
    the entry payload carries only ``ticker``/``lookback_period``; each node
    contributes its own output. ``market_data`` is an internal carrier so the
    single yfinance fetch is shared across nodes.
    """

    ticker: str
    lookback_period: str
    market_data: MarketData
    analyst_report: AnalystReport
    forecast_report: ForecastReport
    judge_verdict: JudgeVerdict


def analyst_node(state: AnalysisState) -> dict:
    """Fetch market data (once) and run the analyst agent."""
    data = fetch_market_data(state["ticker"], period=state.get("lookback_period", "1y"))
    report = run_analyst_agent(data)
    return {"market_data": data, "analyst_report": report}


def forecast_node(state: AnalysisState) -> dict:
    """Run the forecast agent over the already-fetched market data."""
    report = run_forecast_agent(state["market_data"])
    return {"forecast_report": report}


def judge_node(state: AnalysisState) -> dict:
    """Adjudicate the analyst and forecast reports."""
    verdict = run_judge_agent(state["analyst_report"], state["forecast_report"])
    return {"judge_verdict": verdict}


def _build_graph():
    """Compile the analyst -> forecast -> judge -> END state graph."""
    graph = StateGraph(AnalysisState)
    graph.add_node("analyst", analyst_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "forecast")
    graph.add_edge("forecast", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


# Compiled once at import; the graph is stateless across invocations.
_GRAPH = _build_graph()


def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """Run the full analyst -> forecast -> judge pipeline for one ticker.

    Invokes the compiled LangGraph with the request, then merges the node
    outputs into the public :class:`AnalysisResponse`, lifting the headline
    forecast fields to the top level and attaching the judge's verdict.
    """
    final_state: AnalysisState = _GRAPH.invoke(
        {"ticker": request.ticker, "lookback_period": request.lookback_period}
    )

    analyst = final_state["analyst_report"]
    forecast = final_state["forecast_report"]
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
        judge_verdict=judge_verdict,
    )
