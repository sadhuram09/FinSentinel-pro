"""Semantic chunking of 10-K sections into embedding-sized pieces.

Embedding models have a fixed context window and retrieval works best when each
chunk is a single coherent idea, so we split the long Risk Factors / MD&A text
into ~500-token chunks with a 50-token overlap. The overlap keeps a sentence
that straddles a boundary retrievable from either side.

We size chunks in *tokens* (not characters) using the same tokenizer family the
embedding model uses, so "500 tokens" matches what the embedder actually sees.
Every chunk carries its source section in metadata, so a retrieved snippet can
be attributed back to "Risk Factors" vs "MD&A" in the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Same tokenizer family as the embedding model (all-mpnet-base-v2) so token
# counts align with the embedder's context budget.
_TOKENIZER_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_splitter = None


@dataclass
class Chunk:
    """One retrievable unit of text plus provenance metadata."""

    text: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Build (once) a token-aware recursive splitter keyed to the embedder's tokenizer."""
    global _splitter
    if _splitter is None:
        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        _splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )
    return _splitter


def chunk_sections(ticker: str, sections: dict[str, str]) -> list[Chunk]:
    """Split each non-empty section into token-sized, section-tagged chunks.

    Args:
        ticker: The owning ticker (stored in metadata for the collection).
        sections: ``{section_name: raw_text}`` from the SEC loader.

    Returns:
        A flat list of :class:`Chunk`, each tagged with its source section and a
        per-section running index.
    """
    splitter = _get_splitter()
    chunks: list[Chunk] = []
    for section, text in sections.items():
        if not text or not text.strip():
            continue
        for idx, piece in enumerate(splitter.split_text(text)):
            chunks.append(
                Chunk(
                    text=piece,
                    section=section,
                    chunk_index=idx,
                    metadata={"ticker": ticker.upper(), "section": section, "chunk_index": idx},
                )
            )
    return chunks
