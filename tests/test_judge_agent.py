"""Unit tests for the Judge agent's pure logic (no live Groq calls)."""

from __future__ import annotations

import math

import pytest

from backend.agents import judge_agent as J
from backend.schemas import RiskReport, RiskTier, Verdict


# --- consistency score -------------------------------------------------------
def test_consistency_full_agreement():
    # analyst + sentiment both align with the forecast direction -> near 1.0
    score = J._compute_consistency(
        analyst_prob_up=0.80, sentiment_prob_up=0.75, forecast_prob_up=0.78,
        risk_confidence_factor=1.0,
    )
    # 0.6*(1-0.02) + 0.4*(1-0.03) = 0.588 + 0.388 = 0.976
    assert math.isclose(score, 0.976, abs_tol=1e-9)


def test_consistency_disagreement_is_low():
    # analyst + sentiment point down (0.2) while forecast is up (0.8)
    score = J._compute_consistency(
        analyst_prob_up=0.20, sentiment_prob_up=0.20, forecast_prob_up=0.80,
        risk_confidence_factor=1.0,
    )
    # both agreements = 1-0.6 = 0.4 -> 0.6*0.4 + 0.4*0.4 = 0.4
    assert math.isclose(score, 0.40, abs_tol=1e-9)


def test_sentiment_disagreement_lowers_score():
    aligned = J._compute_consistency(0.78, 0.78, 0.78, 1.0)
    opposed = J._compute_consistency(0.78, 0.20, 0.78, 1.0)
    assert opposed < aligned


# --- verdict gating ----------------------------------------------------------
def test_verdict_approved_only_with_high_consistency_and_no_flags():
    assert J._decide_verdict(0.95, []) is Verdict.APPROVED


def test_verdict_any_flag_forces_conflicted_even_with_high_consistency():
    assert J._decide_verdict(0.95, ["low confidence"]) is Verdict.CONFLICTED
    assert J._decide_verdict(0.95, ["wide uncertainty"]) is Verdict.CONFLICTED


def test_verdict_low_consistency_rejected():
    assert J._decide_verdict(0.20, []) is Verdict.REJECTED
    assert J._decide_verdict(0.20, ["low confidence"]) is Verdict.REJECTED  # reject is stricter


def test_verdict_mid_consistency_conflicted():
    assert J._decide_verdict(0.50, []) is Verdict.CONFLICTED


# --- risk confidence factor --------------------------------------------------
def _risk(tier: RiskTier) -> RiskReport:
    return RiskReport(
        ticker="AAPL", lookback_period="1y", var_95=0.02, var_99=0.03, cvar_95=0.03,
        sharpe_ratio=1.0, sortino_ratio=1.0, max_drawdown=0.1, beta=1.0, risk_tier=tier,
    )


def test_risk_factor_high_penalizes_low_and_medium_do_not():
    assert J._risk_confidence_factor(_risk(RiskTier.HIGH)) == 0.85
    assert J._risk_confidence_factor(_risk(RiskTier.MEDIUM)) == 1.0
    assert J._risk_confidence_factor(_risk(RiskTier.LOW)) == 1.0


def test_high_risk_reduces_consistency_by_documented_factor():
    directional = J._compute_consistency(0.80, 0.75, 0.78, 1.0)
    high = J._compute_consistency(0.80, 0.75, 0.78, 0.85)
    assert math.isclose(high, directional * 0.85, abs_tol=1e-9)
    assert high < directional


# --- LLM-output guardrails (pure detectors) ----------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "This forecast is accurate and reliable.",
        "The model has reliable performance.",
        "The prediction was validated and correct.",
    ],
)
def test_banned_vocabulary_detected(text):
    assert J._banned_terms_in(text) != []


def test_clean_text_has_no_banned_terms():
    assert J._banned_terms_in("The agents agree; no flags were raised.") == []


@pytest.mark.parametrize(
    "text",
    ["Low risk increases trust in the call.", "This boosts confidence in the forecast."],
)
def test_risk_boost_phrases_detected(text):
    assert J._risk_boost_phrases_in(text) != []


def test_risk_boost_clean_phrasing_passes():
    assert J._risk_boost_phrases_in("Low risk does not reduce trust in the call.") == []


# --- guardrail integration: banned LLM output -> deterministic fallback -------
def _fake_groq(content: str):
    class _Msg:
        def __init__(self, c): self.content = c

    class _Choice:
        def __init__(self, c): self.message = _Msg(c)

    class _Completions:
        def create(self, **_kw):
            class _R: ...
            r = _R()
            r.choices = [_Choice(content)]
            return r

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, *a, **k): self.chat = _Chat()

    return _Client


def test_guardrail_discards_banned_llm_output(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import groq
    monkeypatch.setattr(groq, "Groq", _fake_groq("This forecast is accurate and reliable."))

    out = J._generate_override_reasoning(
        Verdict.APPROVED, 0.80, [], risk_tier="Low", sentiment_label="neutral",
        evidence_block="Document evidence: INSUFFICIENT.",
    )
    # Banned output is rejected -> deterministic fallback (marked "Groq unavailable").
    assert "accurate" not in out.lower()
    assert "reliable" not in out.lower()
    assert "Groq unavailable" in out


def test_guardrail_discards_risk_boost_llm_output(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import groq
    monkeypatch.setattr(groq, "Groq", _fake_groq("The Low risk tier increases trust in the call."))

    out = J._generate_override_reasoning(
        Verdict.APPROVED, 0.80, [], risk_tier="Low", sentiment_label="neutral",
        evidence_block="Document evidence: INSUFFICIENT.",
    )
    assert "increases trust" not in out.lower()
    assert "Groq unavailable" in out


def test_guardrail_keeps_clean_llm_output(monkeypatch):
    clean = "The verdict reflects strong agreement between the analyst and forecast agents."
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import groq
    monkeypatch.setattr(groq, "Groq", _fake_groq(clean))

    out = J._generate_override_reasoning(
        Verdict.APPROVED, 0.80, [], risk_tier="Low", sentiment_label="neutral",
        evidence_block="Document evidence: INSUFFICIENT.",
    )
    assert out == clean
