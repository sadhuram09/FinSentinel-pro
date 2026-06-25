"""Sentiment agent: reads the recent news narrative around a ticker.

Price/technical signals and news sentiment are largely orthogonal — a stock can
be technically toppy while the headlines turn euphoric, or vice versa. Surfacing
that lets the judge corroborate a forecast (bullish call + bullish coverage) or
flag a divergence (bullish call against a wave of negative news) it would
otherwise miss.

Pipeline: NewsAPI for the last 7 days of headlines -> FinBERT
(ProsusAI/finbert), a BERT model fine-tuned on financial text, classifying each
headline positive/negative/neutral with a confidence score -> aggregate into a
single polarity in [-1, 1].

Both external dependencies degrade gracefully: with no ``NEWSAPI_KEY``, no news,
or transformers/torch unavailable, the agent returns a neutral report with a
``note`` rather than failing the whole analysis.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

from backend.schemas import HeadlineSentiment, SentimentLabel, SentimentReport

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
NEWS_WINDOW_DAYS = 7
MAX_HEADLINES = 50
TOP_K = 3

# Aggregate-polarity thresholds for the discrete label.
POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

# Lazily-initialised FinBERT pipeline (loaded once, reused across requests).
_finbert_pipeline = None
_finbert_unavailable = False


def _get_finbert():
    """Return a cached FinBERT text-classification pipeline, or None if unavailable.

    Loading is deferred to first use because importing torch/transformers and
    downloading the model weights is expensive; we never want it on the import
    path of the web app.
    """
    global _finbert_pipeline, _finbert_unavailable
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    if _finbert_unavailable:
        return None
    try:
        from transformers import pipeline

        _finbert_pipeline = pipeline("text-classification", model=FINBERT_MODEL)
        logger.info("Loaded FinBERT pipeline (%s).", FINBERT_MODEL)
        return _finbert_pipeline
    except Exception:  # noqa: BLE001 - missing torch/transformers or download failure
        logger.exception("FinBERT unavailable; sentiment will degrade to neutral.")
        _finbert_unavailable = True
        return None


def _fetch_headlines(ticker: str, api_key: str) -> list[str]:
    """Fetch up to MAX_HEADLINES English headlines from the last 7 days via NewsAPI."""
    from_date = (datetime.now(timezone.utc) - timedelta(days=NEWS_WINDOW_DAYS)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": ticker,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": MAX_HEADLINES,
            "apiKey": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [a["title"] for a in articles if a.get("title")]


def _signed_score(label: str, confidence: float) -> tuple[SentimentLabel, float]:
    """Map a FinBERT (label, confidence) to a normalised label + signed score.

    FinBERT emits a polarity label and a confidence; we fold them into one signed
    value in [-1, 1] so positive/negative magnitudes are directly comparable and
    can be averaged into an aggregate.
    """
    normalised = label.lower()
    if normalised == "positive":
        return SentimentLabel.POSITIVE, confidence
    if normalised == "negative":
        return SentimentLabel.NEGATIVE, -confidence
    return SentimentLabel.NEUTRAL, 0.0


def _label_from_score(score: float) -> SentimentLabel:
    """Bucket an aggregate polarity into a discrete label."""
    if score >= POSITIVE_THRESHOLD:
        return SentimentLabel.POSITIVE
    if score <= NEGATIVE_THRESHOLD:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def _neutral_report(ticker: str, note: str, headline_count: int = 0) -> SentimentReport:
    """Build a degraded, neutral report carrying an explanatory note."""
    return SentimentReport(
        ticker=ticker.upper(),
        sentiment_score=0.0,
        sentiment_label=SentimentLabel.NEUTRAL,
        headline_count=headline_count,
        top_headlines=[],
        note=note,
    )


def run_sentiment_agent(ticker: str) -> SentimentReport:
    """Fetch recent headlines, score them with FinBERT, and aggregate.

    Returns a neutral report with a ``note`` (never raises) when news or the model
    is unavailable, so a sentiment outage cannot take down the whole pipeline.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return _neutral_report(ticker, "NEWSAPI_KEY not set; sentiment skipped.")

    try:
        headlines = _fetch_headlines(ticker, api_key)
    except Exception:  # noqa: BLE001 - network/HTTP/quota errors
        logger.exception("NewsAPI fetch failed for %s.", ticker)
        return _neutral_report(ticker, "NewsAPI request failed; sentiment skipped.")

    if not headlines:
        return _neutral_report(ticker, "No headlines found in the last 7 days.")

    pipe = _get_finbert()
    if pipe is None:
        return _neutral_report(
            ticker, "FinBERT unavailable (transformers/torch).", headline_count=len(headlines)
        )

    results = pipe(headlines)  # one {label, score} per headline

    scored: list[HeadlineSentiment] = []
    for text, res in zip(headlines, results):
        label, signed = _signed_score(res["label"], float(res["score"]))
        scored.append(HeadlineSentiment(headline=text, label=label, score=signed))

    aggregate = sum(h.score for h in scored) / len(scored)
    top = sorted(scored, key=lambda h: abs(h.score), reverse=True)[:TOP_K]

    return SentimentReport(
        ticker=ticker.upper(),
        sentiment_score=aggregate,
        sentiment_label=_label_from_score(aggregate),
        headline_count=len(scored),
        top_headlines=top,
        note=None,
    )
