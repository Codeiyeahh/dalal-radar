"""
Embedder Module.

Loads a sentence-transformer model and generates vector embeddings
for text chunks produced by the chunker.

The model is cached in a module-level singleton so repeated calls
never reload from disk.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache for the embedding model
_model_cache: Optional[object] = None


# ---------------------------------------------------------------------------
# Model loader (cached singleton)
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    Load and return the all-MiniLM-L6-v2 SentenceTransformer model.
    Cached after first call.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    logger.info("Loading embedding model '%s' …", _MODEL_NAME)

    from sentence_transformers import SentenceTransformer
    _model_cache = SentenceTransformer(_MODEL_NAME)

    logger.info("Embedding model '%s' loaded successfully", _MODEL_NAME)
    return _model_cache


# ---------------------------------------------------------------------------
# Embed chunks
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an "embedding" key (list[float], 384-d) to each chunk dict.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have a "text" key.

    Returns
    -------
    list[dict]
        Same dicts enriched with "embedding".
    """
    if not chunks:
        logger.warning("embed_chunks called with empty list — returning []")
        return []

    model = get_embedding_model()
    texts = [c.get("text", "") for c in chunks]
    logger.info("Embedding %d chunk(s) …", len(texts))

    embeddings = model.encode(texts, show_progress_bar=False)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    logger.info(
        "embed_chunks -> %d chunk(s) embedded (dim=%d)",
        len(chunks), len(chunks[0]["embedding"]),
    )
    return chunks


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Embedder Module Smoke Test")
    print("=" * 70)

    sample_chunks = [
        {"text": "Reliance Industries is showing bullish momentum.", "chunk_id": "a1"},
        {"text": "HDFC Bank consolidating near support levels.", "chunk_id": "a2"},
        {"text": "Infosys under pressure from global IT slowdown.", "chunk_id": "a3"},
    ]

    print(f"\n[1] Embedding {len(sample_chunks)} sample chunk(s) …\n")
    enriched = embed_chunks(sample_chunks)

    for c in enriched:
        emb = c["embedding"]
        print(f"    chunk_id={c['chunk_id']}  |  dim={len(emb)}  |  first 5: {[round(v, 4) for v in emb[:5]]}")

    print("\n" + "=" * 70)
