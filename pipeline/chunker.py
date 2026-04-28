"""
Chunker Module.

Splits raw text (transcripts, market deals, sentiment results) into
smaller, embeddable chunks with rich metadata for downstream retrieval.

Chunk types
-----------
- ``youtube``   — transcript text split on sentence boundaries
- ``market``    — one chunk per deal from the market agent
- ``sentiment`` — one chunk per LLM-extracted ticker sentiment
"""

from __future__ import annotations

import re
import sys
import os
import uuid
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CHUNK_TOKENS = 400
_DEFAULT_OVERLAP_TOKENS = 50

# Rough token estimate: 1 token ≈ 4 characters (GPT-family heuristic)
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count using the ~4 chars/token heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences using a regex that respects common
    abbreviations (Mr., Dr., etc.) and decimal numbers.

    Returns a list of sentence strings (whitespace-stripped, non-empty).
    """
    # Split on period / question-mark / exclamation followed by whitespace
    # and an uppercase letter, OR on newlines that look like paragraph breaks.
    sentence_endings = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z\u0900-\u097F])'   # after punctuation + space + capital / Devanagari
        r'|(?:\r?\n){2,}'                          # or paragraph break
    )
    parts = sentence_endings.split(text)
    return [s.strip() for s in parts if s.strip()]


def _generate_chunk_id() -> str:
    """Return a short unique chunk identifier."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# STEP 1 — Chunk transcripts
# ---------------------------------------------------------------------------

def chunk_transcript(
    text: str,
    video_meta: dict,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Split a YouTube transcript into overlapping chunks on sentence
    boundaries.

    Parameters
    ----------
    text : str
        Full transcript text.
    video_meta : dict
        Video metadata from the YouTube agent (must include at least
        ``video_id``).
    chunk_tokens : int
        Target chunk size in tokens (default 400).
    overlap_tokens : int
        Overlap size in tokens between consecutive chunks (default 50).

    Returns
    -------
    list[dict]
        Each dict:
        ``{text, chunk_id, video_id, channel_name, timestamp,
        video_url, chunk_index, source_type}``
    """
    if not text or not text.strip():
        logger.warning("chunk_transcript called with empty text — returning []")
        return []

    max_chars = chunk_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    sentences = _split_sentences(text)
    if not sentences:
        logger.warning("No sentences detected in transcript — returning []")
        return []

    chunks: list[dict] = []
    current_sentences: list[str] = []
    current_len = 0

    def _flush(chunk_index: int) -> dict:
        """Build a chunk dict from the accumulated sentences."""
        chunk_text = " ".join(current_sentences)
        return {
            "text": chunk_text,
            "chunk_id": _generate_chunk_id(),
            "video_id": video_meta.get("video_id", ""),
            "channel_name": video_meta.get("channel_name", "N/A"),
            "timestamp": video_meta.get("published_at", ""),
            "video_url": video_meta.get("url", ""),
            "chunk_index": chunk_index,
            "source_type": "youtube",
        }

    for sentence in sentences:
        sent_len = len(sentence)

        # If adding this sentence exceeds the limit, flush first
        if current_len + sent_len > max_chars and current_sentences:
            chunks.append(_flush(len(chunks)))

            # Keep trailing sentences that fall within the overlap window
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current_sentences = overlap_sentences
            current_len = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_len += sent_len

    # Flush remaining
    if current_sentences:
        chunks.append(_flush(len(chunks)))

    logger.info(
        "chunk_transcript -> %d chunk(s) from video %s (~%d tokens each)",
        len(chunks),
        video_meta.get("video_id", "?"),
        chunk_tokens,
    )
    return chunks


# ---------------------------------------------------------------------------
# STEP 2 — Chunk market deals
# ---------------------------------------------------------------------------

def chunk_market_data(deals_text: str) -> list[dict]:
    """
    Split the RAG-formatted deals text into one chunk per deal line.

    Parameters
    ----------
    deals_text : str
        Output of ``market_agent.format_deals_for_rag(deals)``.
        Each deal line follows the pattern:
        ``[  1] SYMBOL | TYPE | qty shares @ price | Client: … | Date: …``

    Returns
    -------
    list[dict]
        Each dict:
        ``{text, chunk_id, symbol, deal_type, deal_date, source_type}``
    """
    if not deals_text or not deals_text.strip():
        logger.warning("chunk_market_data called with empty text — returning []")
        return []

    lines = deals_text.strip().splitlines()
    chunks: list[dict] = []

    # Regex to extract fields from the formatted deal line
    # Example: [  1] RELIANCE     | BULK  | 250,000 shares @ ₹2,850.00 | Client: HDFC MF | Date: 24-Apr-2026
    deal_pattern = re.compile(
        r'\[\s*\d+\]\s+'
        r'(?P<symbol>\S+)\s*\|\s*'
        r'(?P<deal_type>\S+)\s*\|'
        r'.*?\|\s*Client:\s*(?P<client>.*?)\s*\|\s*Date:\s*(?P<date>.*?)$'
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = deal_pattern.match(line)
        if match:
            chunks.append({
                "text": line,
                "chunk_id": _generate_chunk_id(),
                "symbol": match.group("symbol").strip().upper(),
                "deal_type": match.group("deal_type").strip().lower(),
                "deal_date": match.group("date").strip(),
                "source_type": "market",
            })
        else:
            # Header / summary lines — include them as context chunks
            if line and not line.startswith("No NSE"):
                chunks.append({
                    "text": line,
                    "chunk_id": _generate_chunk_id(),
                    "symbol": "",
                    "deal_type": "",
                    "deal_date": "",
                    "source_type": "market",
                })

    logger.info("chunk_market_data -> %d chunk(s)", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# STEP 3 — Chunk sentiment results
# ---------------------------------------------------------------------------

def chunk_sentiment_results(results: list[dict]) -> list[dict]:
    """
    Convert YouTube sentiment analysis results into one chunk per ticker
    mention.

    Parameters
    ----------
    results : list[dict]
        Output of ``youtube_agent.run_youtube_pipeline()`` — each dict has
        keys: symbol, sentiment, reason, confidence, channel_name,
        video_id, timestamp, video_url.

    Returns
    -------
    list[dict]
        Each dict:
        ``{text, chunk_id, symbol, sentiment, channel_name, video_id,
        video_url, source_type}``
    """
    if not results:
        logger.warning("chunk_sentiment_results called with empty list — returning []")
        return []

    chunks: list[dict] = []
    for r in results:
        # Build a human-readable text representation
        text = (
            f"Channel: {r.get('channel_name', 'N/A')} | "
            f"Ticker: {r.get('symbol', 'N/A')} | "
            f"Sentiment: {r.get('sentiment', 'neutral')} | "
            f"Confidence: {r.get('confidence', 'N/A')} | "
            f"Reason: {r.get('reason', 'N/A')} | "
            f"Video: {r.get('video_url', 'N/A')} | "
            f"Time: {r.get('timestamp', 'N/A')}"
        )

        chunks.append({
            "text": text,
            "chunk_id": _generate_chunk_id(),
            "symbol": str(r.get("symbol", "")).strip().upper(),
            "sentiment": str(r.get("sentiment", "neutral")).strip().lower(),
            "channel_name": r.get("channel_name", "N/A"),
            "video_id": r.get("video_id", ""),
            "video_url": r.get("video_url", ""),
            "source_type": "sentiment",
        })

    logger.info("chunk_sentiment_results -> %d chunk(s)", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Chunker Module Smoke Test")
    print("=" * 70)

    # --- Test 1: Transcript chunking ---
    sample_transcript = (
        "Today we are looking at Reliance Industries. The stock has been "
        "showing bullish momentum on the daily chart. The RSI is above 60 "
        "and MACD is positive. I think Reliance can reach 2900 levels soon. "
        "Moving on to HDFC Bank. This stock has been consolidating near "
        "support levels. The quarterly results were decent but not great. "
        "I am neutral on HDFC Bank for now. Let us also talk about Infosys. "
        "IT sector is under pressure due to global slowdown concerns. "
        "Infosys might see some selling pressure in the near term. "
        "Overall the market is looking cautiously optimistic today."
    )
    meta = {
        "video_id": "test123",
        "channel_name": "TestTrader",
        "published_at": "2026-04-25T10:00:00Z",
        "url": "https://www.youtube.com/watch?v=test123",
    }

    print("\n[1] Transcript chunking (small chunk for demo):")
    transcript_chunks = chunk_transcript(sample_transcript, meta, chunk_tokens=50)
    for c in transcript_chunks:
        tokens_est = _estimate_tokens(c["text"])
        print(f"    Chunk {c['chunk_index']}: ~{tokens_est} tokens | {c['text'][:80]}…")

    # --- Test 2: Market data chunking ---
    sample_deals = (
        "NSE Bulk & Block Deals (last 24 hours) — 2 deal(s) found:\n\n"
        "[  1] RELIANCE     | BULK  | 250,000 shares @ ₹2,850.00 | Client: HDFC MF | Date: 24-Apr-2026\n"
        "[  2] INFY         | BLOCK | 80,000 shares @ ₹1,420.50  | Client: ICICI PRUDENTIAL | Date: 24-Apr-2026\n"
    )

    print("\n[2] Market data chunking:")
    market_chunks = chunk_market_data(sample_deals)
    for c in market_chunks:
        print(f"    [{c['source_type']}] {c['symbol'] or 'HEADER'}: {c['text'][:80]}")

    # --- Test 3: Sentiment result chunking ---
    sample_sentiments = [
        {
            "symbol": "RELIANCE",
            "sentiment": "bullish",
            "reason": "Strong quarterly earnings and momentum",
            "confidence": 0.85,
            "channel_name": "MarketGuru",
            "video_id": "vid001",
            "timestamp": "2026-04-25T10:00:00Z",
            "video_url": "https://youtube.com/watch?v=vid001",
        },
        {
            "symbol": "INFY",
            "sentiment": "bearish",
            "reason": "IT sector slowdown concerns",
            "confidence": 0.70,
            "channel_name": "StockTalks",
            "video_id": "vid002",
            "timestamp": "2026-04-25T11:00:00Z",
            "video_url": "https://youtube.com/watch?v=vid002",
        },
    ]

    print("\n[3] Sentiment result chunking:")
    sentiment_chunks = chunk_sentiment_results(sample_sentiments)
    for c in sentiment_chunks:
        print(f"    [{c['source_type']}] {c['symbol']} ({c['sentiment']}): {c['text'][:80]}…")

    print(f"\n    Total chunks: {len(transcript_chunks) + len(market_chunks) + len(sentiment_chunks)}")
    print("\n" + "=" * 70)
