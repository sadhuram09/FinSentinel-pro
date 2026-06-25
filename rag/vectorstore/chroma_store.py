"""ChromaDB-backed vector store, one collection per ticker.

Each ticker's 10-K chunks live in their own persistent collection so retrieval
is scoped to the right company and a re-ingest of one name never disturbs
another. Embeddings use sentence-transformers/all-mpnet-base-v2 — a strong
general-purpose semantic model — and the collection is configured for cosine
distance so a chunk's relevance maps cleanly to ``similarity = 1 - distance``.

The index persists to ``rag_store/`` (gitignored, like trained models: it is a
large, regenerable artifact, not source).
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from ftfy import fix_text

from rag.ingestion.chunker import Chunk

# Project-root rag_store/. chroma_store.py is at rag/vectorstore/, so parents[2]
# is the project root.
STORE_DIR = Path(__file__).resolve().parents[2] / "rag_store"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

_client = None
_embedding_fn = None


def _get_client() -> "chromadb.ClientAPI":
    """Return a cached persistent Chroma client rooted at STORE_DIR."""
    global _client
    if _client is None:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(STORE_DIR))
    return _client


def _get_embedding_fn():
    """Return a cached sentence-transformers embedding function (loads model once)."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    return _embedding_fn


def _collection_name(ticker: str) -> str:
    return f"tenk_{ticker.lower()}"


def get_collection(ticker: str):
    """Get-or-create the cosine-distance collection for a ticker."""
    return _get_client().get_or_create_collection(
        name=_collection_name(ticker),
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def delete_ticker(ticker: str) -> None:
    """Drop a ticker's collection so it can be re-ingested from scratch."""
    try:
        _get_client().delete_collection(_collection_name(ticker))
    except Exception:  # noqa: BLE001 - collection may not exist; nothing to drop
        pass


def has_documents(ticker: str) -> bool:
    """True if the ticker's collection already holds chunks (skip re-ingest)."""
    try:
        return get_collection(ticker).count() > 0
    except Exception:  # noqa: BLE001 - treat any access error as "not ingested"
        return False


def upsert_chunks(ticker: str, chunks: list[Chunk]) -> int:
    """Embed and upsert chunks into the ticker's collection; return the count.

    IDs are deterministic (``{ticker}-{section}-{idx}``) so re-ingesting the same
    filing overwrites rather than duplicates.
    """
    if not chunks:
        return 0
    collection = get_collection(ticker)
    ids = [f"{ticker.upper()}-{c.section}-{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query(ticker: str, query_text: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks for a query as dicts with a cosine similarity score.

    ``similarity = 1 - cosine_distance`` lies in [0, 1] for these normalised
    embeddings, where 1.0 is an exact semantic match.
    """
    collection = get_collection(ticker)
    n = min(k, collection.count())
    if n == 0:
        return []
    res = collection.query(query_texts=[query_text], n_results=n)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    results = []
    for doc, meta, dist in zip(docs, metas, distances):
        results.append(
            {
                "section": meta.get("section", "unknown"),
                # Repair mojibake on read: Chroma's native layer can decode stored
                # UTF-8 via the host ANSI codepage (e.g. cp1252 under uvicorn on
                # Windows), turning ' into "â€™". ftfy reverses that deterministic
                # double-encoding and is a no-op on already-clean text.
                "snippet": fix_text(doc),
                "similarity_score": round(1.0 - float(dist), 4),
            }
        )
    return results
