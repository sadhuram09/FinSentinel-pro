"""Retrieval entry point: query a ticker's 10-K and gauge evidence strength.

This ties the pipeline together: it lazily ingests the latest 10-K the first
time a ticker is queried (fetch -> chunk -> embed -> persist), then answers
queries from the persisted Chroma collection.

The important nuance is *retrieval confidence*. A vector store always returns
its top-k nearest chunks — even when none are actually relevant. So we expose
the top chunk's similarity and bucket it into a ``confidence_tier`` (strong /
moderate / insufficient). That lets the judge match its language to the actual
evidence quality — cite confidently, hedge as partial, or declare insufficient
— instead of treating every retrieval as equally trustworthy.

Returns a plain dataclass so this package stays independent of the web app's
Pydantic schemas; the orchestrator maps it into the API response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.ingestion.chunker import chunk_sections
from rag.ingestion.sec_loader import load_latest_10k
from rag.vectorstore import chroma_store

logger = logging.getLogger(__name__)

TOP_K = 5
# Tier thresholds on the top chunk's cosine similarity. Real 10-K matches for a
# relevant query cluster around 0.55-0.65, while off-topic queries score ~0.05,
# so these bands separate solid support from loosely-related from irrelevant.
STRONG_THRESHOLD = 0.60
MODERATE_THRESHOLD = 0.45

# Tier labels (kept as plain strings so this package stays independent of the
# web app's enums; the orchestrator maps them to RetrievalTier).
TIER_STRONG = "strong"
TIER_MODERATE = "moderate"
TIER_INSUFFICIENT = "insufficient"


def classify_tier(top_similarity: float) -> str:
    """Map the top-chunk similarity onto a strong / moderate / insufficient tier."""
    if top_similarity >= STRONG_THRESHOLD:
        return TIER_STRONG
    if top_similarity >= MODERATE_THRESHOLD:
        return TIER_MODERATE
    return TIER_INSUFFICIENT


@dataclass
class RetrievalResult:
    """Top-k retrieved evidence plus a tiered read on its quality."""

    query: str
    evidence: list[dict] = field(default_factory=list)  # {section, snippet, similarity_score}
    retrieval_confidence: float = 0.0  # top chunk's similarity (0.0 if nothing retrieved)
    confidence_tier: str = TIER_INSUFFICIENT
    note: str | None = None


def ensure_ingested(ticker: str) -> str | None:
    """Ingest the ticker's latest 10-K if not already indexed.

    Returns an error note on failure (so retrieval degrades to "no evidence"
    rather than crashing the pipeline), or None on success.
    """
    if chroma_store.has_documents(ticker):
        return None
    try:
        tenk = load_latest_10k(ticker)
        chunks = chunk_sections(ticker, tenk.sections)
        if not chunks:
            return "10-K fetched but no Risk Factors / MD&A text could be extracted."
        count = chroma_store.upsert_chunks(ticker, chunks)
        logger.info("Ingested %d 10-K chunks for %s (%s).", count, ticker, tenk.accession)
        return None
    except Exception as exc:  # noqa: BLE001 - SEC/network/parse failure
        logger.exception("10-K ingestion failed for %s.", ticker)
        return f"10-K ingestion failed: {exc}"


def retrieve(ticker: str, query: str, k: int = TOP_K) -> RetrievalResult:
    """Return the top-k 10-K chunks for ``query`` with a retrieval-confidence read."""
    note = ensure_ingested(ticker)
    if note is not None:
        return RetrievalResult(query=query, note=note)

    evidence = chroma_store.query(ticker, query, k=k)
    if not evidence:
        return RetrievalResult(query=query, note="No matching 10-K chunks found.")

    top_similarity = evidence[0]["similarity_score"]
    tier = classify_tier(top_similarity)
    notes = {
        TIER_STRONG: None,
        TIER_MODERATE: "Top match is moderate; evidence is partial / directionally relevant.",
        TIER_INSUFFICIENT: "Top match below the moderate threshold; evidence is insufficient.",
    }
    return RetrievalResult(
        query=query,
        evidence=evidence,
        retrieval_confidence=top_similarity,
        confidence_tier=tier,
        note=notes[tier],
    )
