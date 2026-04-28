"""
Vector Store Module.

ChromaDB wrapper for storing and retrieving embedded chunks.

Functions
---------
get_collection()                       -> chromadb.Collection
store_chunks(chunks)                   -> None
search(query, n_results, filters)      -> list[dict]
get_by_symbol(symbol, n_results)       -> list[dict]
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import chromadb

# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHROMA_PERSIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db",
)
_COLLECTION_NAME = "dalal_radar"

# Module-level cache
_collection_cache = None


# ---------------------------------------------------------------------------
# Collection accessor
# ---------------------------------------------------------------------------

def get_collection():
    """
    Initialise a ChromaDB persistent client and return (or create) the
    ``dalal_radar`` collection.

    The collection is cached after the first call.
    """
    global _collection_cache

    if _collection_cache is not None:
        return _collection_cache

    logger.info(
        "Initialising ChromaDB (persist_directory=%s) …", _CHROMA_PERSIST_PATH
    )
    client = chromadb.PersistentClient(path=_CHROMA_PERSIST_PATH)
    _collection_cache = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (existing docs: %d)",
        _COLLECTION_NAME,
        _collection_cache.count(),
    )
    return _collection_cache


# ---------------------------------------------------------------------------
# Store chunks
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict]) -> None:
    """
    Upsert embedded chunks into ChromaDB.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must contain ``chunk_id``, ``text``, and ``embedding``.
        All other keys are stored as metadata.
    """
    if not chunks:
        logger.warning("store_chunks called with empty list — nothing to store")
        return

    collection = get_collection()

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        if not chunk_id:
            logger.warning("Skipping chunk without chunk_id")
            continue

        ids.append(chunk_id)
        documents.append(chunk.get("text", ""))
        embeddings.append(chunk.get("embedding", []))

        # Build metadata — everything except text, embedding, chunk_id
        meta = {
            k: v
            for k, v in chunk.items()
            if k not in ("text", "embedding", "chunk_id")
            and isinstance(v, (str, int, float, bool))
        }
        metadatas.append(meta)

    if not ids:
        logger.warning("No valid chunks to store after filtering")
        return

    # ChromaDB upsert handles duplicates gracefully
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info("store_chunks -> %d chunk(s) upserted into ChromaDB", len(ids))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    n_results: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Embed *query* and search ChromaDB by cosine similarity.

    Parameters
    ----------
    query : str
        Natural-language search query.
    n_results : int
        Maximum results to return (default 5).
    filters : dict, optional
        ChromaDB metadata filters, e.g. ``{"symbol": "RELIANCE"}``.

    Returns
    -------
    list[dict]
        Each dict: ``{text, metadata, distance}`` sorted by relevance.
    """
    # Import embedder lazily to avoid circular imports
    from pipeline.embedder import get_embedding_model

    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()

    collection = get_collection()

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count() or n_results),
    }
    if filters:
        # Wrap simple key-value filters into ChromaDB $eq format
        where_clause: dict = {}
        if len(filters) == 1:
            key, val = next(iter(filters.items()))
            where_clause = {key: val}
        else:
            where_clause = {
                "$and": [{k: v} for k, v in filters.items()]
            }
        query_kwargs["where"] = where_clause

    # Guard against querying an empty collection
    if collection.count() == 0:
        logger.warning("search called on empty collection — returning []")
        return []

    results = collection.query(**query_kwargs)

    # Unpack ChromaDB's nested list format
    output: list[dict] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        output.append({
            "text": text,
            "metadata": meta,
            "distance": dist,
        })

    logger.info(
        "search('%s') -> %d result(s) (filters=%s)",
        query[:50], len(output), filters,
    )
    return output


# ---------------------------------------------------------------------------
# Convenience: search by symbol
# ---------------------------------------------------------------------------

def get_by_symbol(symbol: str, n_results: int = 10) -> list[dict]:
    """
    Retrieve all chunks mentioning a specific ticker symbol.

    Parameters
    ----------
    symbol : str
        NSE/BSE ticker, e.g. ``"RELIANCE"``.
    n_results : int
        Max results (default 10).

    Returns
    -------
    list[dict]
        Same format as ``search()``.
    """
    symbol_upper = symbol.strip().upper()
    logger.info("get_by_symbol('%s', n=%d)", symbol_upper, n_results)
    return search(
        query=f"{symbol_upper} stock analysis",
        n_results=n_results,
        filters={"symbol": symbol_upper},
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Vector Store Smoke Test")
    print("=" * 70)

    # Create sample chunks with embeddings
    from pipeline.embedder import embed_chunks

    sample = [
        {
            "text": "Reliance Industries bullish on daily chart.",
            "chunk_id": "smoke_01",
            "symbol": "RELIANCE",
            "source_type": "sentiment",
            "sentiment": "bullish",
        },
        {
            "text": "HDFC Bank neutral, consolidating near support.",
            "chunk_id": "smoke_02",
            "symbol": "HDFCBANK",
            "source_type": "sentiment",
            "sentiment": "neutral",
        },
        {
            "text": "Infosys bearish outlook due to IT slowdown.",
            "chunk_id": "smoke_03",
            "symbol": "INFY",
            "source_type": "sentiment",
            "sentiment": "bearish",
        },
    ]

    print("\n[1] Embedding sample chunks …")
    embedded = embed_chunks(sample)

    print("[2] Storing in ChromaDB …")
    store_chunks(embedded)

    print(f"[3] Collection count: {get_collection().count()}")

    print("\n[4] Searching: 'bullish stocks' …")
    hits = search("bullish stocks", n_results=3)
    for h in hits:
        print(f"    dist={h['distance']:.4f} | {h['text']}")

    print("\n[5] get_by_symbol('RELIANCE') …")
    sym_hits = get_by_symbol("RELIANCE")
    for h in sym_hits:
        print(f"    dist={h['distance']:.4f} | {h['text']}")

    print("\n" + "=" * 70)
