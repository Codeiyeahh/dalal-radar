"""
Chat Agent Module.

RAG chatbot that answers questions about NSE/BSE sentiment with
citations and trading recommendations.

Pipeline per query
------------------
1. Extract ticker symbols from the user query  (LLM)
2. Retrieve relevant chunks from ChromaDB      (vector_store)
3. Build a structured context string
4. Generate a grounded answer                   (LLM)
5. Extract cited sources from the response
6. Optionally generate a trading recommendation (LLM)
"""

from __future__ import annotations

import json
import re
import sys
import os
from collections import Counter
from typing import Optional
import time


import httpx

import sys
import os

# ---------------------------------------------------------------------------
# Bootstrap path so we can import agents and config when run as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure tools are registered with Hermes
import agents.market_agent
import agents.youtube_agent
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, FALLBACK_MODELS, logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OPENROUTER_URL = OPENROUTER_BASE_URL
_LLM_MODEL = OPENROUTER_MODEL

_TICKER_EXTRACT_PROMPT = (
    "Extract NSE/BSE stock ticker symbols from this query. "
    "Return JSON only: {tickers: [list of symbols in caps]}"
)

_SYSTEM_PROMPT = (
    "You are DalalRadar, an AI assistant for Indian stock market analysis. "
    "Answer ONLY using the context provided below. "
    "Every claim you make MUST be backed by the context. "
    "If the context is insufficient, say so clearly. "
    "Always cite your sources using [Channel Name, timestamp/URL]. "
    "Never hallucinate tickers, prices or sentiments not in the context. "
    "Be concise and factual."
)

_RECOMMENDATION_PROMPT_TEMPLATE = (
    "Based on this analysis, give a brief trading recommendation "
    "for {symbol}. Include confidence level (low/medium/high) based on "
    "number of sources. Always add disclaimer: Not financial advice."
)

# Minimum number of agreeing sources before a recommendation is generated
_MIN_SOURCES_FOR_REC = 3

def _call_llm(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """
    Send a chat-completion request to OpenRouter with fallback model support.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    # Rate limit handling: 8 second sleep between calls
    time.sleep(8)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dalal-radar",
        "X-Title": "Dalal Radar",
    }

    # Build the model list to try: [Primary, Fallback 1, Fallback 2, ...]
    models_to_try = [OPENROUTER_MODEL]
    for fm in FALLBACK_MODELS:
        if fm not in models_to_try:
            models_to_try.append(fm)

    endpoint = OPENROUTER_BASE_URL
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint.rstrip("/") + "/chat/completions"

    last_error = None
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            with httpx.Client(timeout=60) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                
                # Try fallback on any 4xx error
                if 400 <= resp.status_code < 500:
                    logger.warning("Model %s returned %d. Trying fallback...", model, resp.status_code)
                    continue
                
                resp.raise_for_status()

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                logger.info("Using model: %s", model)
                return content.strip()
            
        except Exception as e:
            last_error = e
            logger.warning("Model %s failed: %s. Trying next...", model, e)

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def _call_llm_with_retry(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> str:
    """
    Wrapper for _call_llm that adds exponential backoff specifically for 429 (Rate Limit) errors.
    """
    for attempt in range(max_retries):
        try:
            return _call_llm(messages, temperature, max_tokens)
        except Exception as e:
            # Check for 429 in error message or response code if available
            if "429" in str(e):
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s, 100s
                logger.warning("Rate limited (429), waiting %ds before retry (attempt %d/%d)...", 
                               wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                # Re-raise other errors immediately
                raise e
    
    raise RuntimeError(f"Max retries exceeded for LLM call after {max_retries} attempts due to rate limiting.")


# ---------------------------------------------------------------------------
# STEP 1 — Extract tickers from query
# ---------------------------------------------------------------------------

def _extract_tickers(query: str) -> list[str]:
    """
    Use the LLM to pull NSE/BSE ticker symbols out of a natural-language
    query.  Returns an uppercase list, e.g. ``["RELIANCE", "INFY"]``.
    """
    logger.info("STEP 1 — Extracting tickers from query …")

    try:
        raw = _call_llm_with_retry([
            {"role": "system", "content": _TICKER_EXTRACT_PROMPT},
            {"role": "user", "content": query},
        ])

        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

        parsed = json.loads(cleaned)
        tickers = parsed.get("tickers", [])
        tickers = [t.strip().upper() for t in tickers if t.strip()]

        logger.info("  Tickers extracted: %s", tickers or "(none)")
        return tickers

    except Exception as exc:
        logger.warning("  Ticker extraction failed (%s) — proceeding without", exc)
        return []


# ---------------------------------------------------------------------------
# STEP 2 — Retrieve context
# ---------------------------------------------------------------------------

def _retrieve_context(query: str, tickers: list[str]) -> list[dict]:
    """
    Fetch relevant chunks from ChromaDB.  Combines per-ticker lookups
    with a general semantic search and deduplicates.
    """
    from retrieval.vector_store import search, get_by_symbol

    logger.info("STEP 2 — Retrieving context from vector store …")

    all_results: list[dict] = []
    seen_texts: set[str] = set()

    # Per-ticker retrieval
    for ticker in tickers:
        hits = get_by_symbol(ticker, n_results=8)
        for h in hits:
            if h["text"] not in seen_texts:
                seen_texts.add(h["text"])
                all_results.append(h)

    # General semantic search
    general_hits = search(query, n_results=8)
    for h in general_hits:
        if h["text"] not in seen_texts:
            seen_texts.add(h["text"])
            all_results.append(h)

    logger.info("  Retrieved %d unique chunk(s)", len(all_results))
    return all_results


# ---------------------------------------------------------------------------
# STEP 3 — Build context string
# ---------------------------------------------------------------------------

def _build_context(results: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """
    Format retrieved chunks into a structured context string for the LLM.

    Returns
    -------
    (context_string, youtube_chunks, market_chunks)
    """
    logger.info("STEP 3 — Building context string …")

    youtube_chunks: list[dict] = []
    market_chunks: list[dict] = []
    sentiment_chunks: list[dict] = []

    for r in results:
        meta = r.get("metadata", {})
        source_type = meta.get("source_type", "")

        if source_type == "youtube":
            youtube_chunks.append(r)
        elif source_type == "market":
            market_chunks.append(r)
        elif source_type == "sentiment":
            sentiment_chunks.append(r)
        else:
            # Unknown source type — still include it
            youtube_chunks.append(r)

    lines: list[str] = []

    # YouTube / transcript chunks
    for c in youtube_chunks:
        meta = c.get("metadata", {})
        lines.append(
            f"[YouTube] Channel: {meta.get('channel_name', 'N/A')} | "
            f"Ticker: {meta.get('symbol', 'N/A')} | "
            f"Sentiment: {meta.get('sentiment', 'N/A')} | "
            f"Content: {c['text'][:300]} | "
            f"Source: {meta.get('video_url', 'N/A')}"
        )

    # Sentiment chunks (from LLM analysis)
    for c in sentiment_chunks:
        meta = c.get("metadata", {})
        lines.append(
            f"[YouTube] Channel: {meta.get('channel_name', 'N/A')} | "
            f"Ticker: {meta.get('symbol', 'N/A')} | "
            f"Sentiment: {meta.get('sentiment', 'N/A')} | "
            f"Reason: {c['text'][:300]} | "
            f"Source: {meta.get('video_url', 'N/A')}"
        )

    # Market deal chunks
    for c in market_chunks:
        meta = c.get("metadata", {})
        lines.append(
            f"[Market] Symbol: {meta.get('symbol', 'N/A')} | "
            f"Type: {meta.get('deal_type', 'N/A')} | "
            f"Details: {c['text'][:300]} | "
            f"Date: {meta.get('deal_date', 'N/A')}"
        )

    context_str = "\n".join(lines)
    logger.info(
        "  Context built: %d YouTube, %d sentiment, %d market chunk(s)",
        len(youtube_chunks), len(sentiment_chunks), len(market_chunks),
    )
    return context_str, youtube_chunks + sentiment_chunks, market_chunks


from temp_hermes.run_agent import AIAgent

_agent_instance = None

def get_agent() -> AIAgent:
    """Singleton pattern to maintain conversation state across queries."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AIAgent(
            model=_LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=_OPENROUTER_URL,
            provider="openrouter",
            enabled_toolsets=["market", "youtube"],
            save_trajectories=True,
        )
    return _agent_instance

# ---------------------------------------------------------------------------
# STEP 4 — Generate answer
# ---------------------------------------------------------------------------

def _generate_answer(
    query: str,
    context_str: str,
    conversation_history: list[dict],
) -> str:
    """Call the LLM with system prompt, conversation history, and context."""
    logger.info("STEP 4 — Generating grounded answer …")

    # Combine original system prompt with the dynamically retrieved context
    system_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"CONTEXT (from live data sources — last 24 hours):\n"
        f"{'─' * 50}\n"
        f"{context_str}\n"
        f"{'─' * 50}"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})

    try:
        answer = _call_llm_with_retry(messages)
    except Exception as exc:
        logger.error("  Answer generation failed: %s", exc)
        answer = "I'm sorry, I encountered an error generating the answer."

    logger.info("  Answer generated (%d chars)", len(answer))
    return answer


# ---------------------------------------------------------------------------
# STEP 5 — Extract citations
# ---------------------------------------------------------------------------

def _extract_citations(
    answer: str,
    yt_chunks: list[dict],
    mkt_chunks: list[dict],
) -> list[dict]:
    """
    Parse the LLM response and match cited channel names / URLs back
    to the retrieved chunks.
    """
    logger.info("STEP 5 — Extracting citations …")

    citations: list[dict] = []
    seen: set[str] = set()

    all_chunks = yt_chunks + mkt_chunks
    for chunk in all_chunks:
        meta = chunk.get("metadata", {})
        channel = meta.get("channel_name", "")
        video_url = meta.get("video_url", "")
        symbol = meta.get("symbol", "")

        # Check if the channel name or URL appears in the answer
        mentioned = False
        if channel and channel != "N/A" and channel.lower() in answer.lower():
            mentioned = True
        if video_url and video_url != "N/A" and video_url in answer:
            mentioned = True
        # Also count if the symbol is mentioned
        if symbol and symbol != "N/A" and symbol in answer:
            mentioned = True

        if mentioned:
            dedup_key = f"{channel}|{symbol}|{video_url}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                citations.append({
                    "channel": channel or "N/A",
                    "video_url": video_url or "N/A",
                    "sentiment": meta.get("sentiment", "N/A"),
                    "symbol": symbol or "N/A",
                })

    logger.info("  Citations found: %d", len(citations))
    return citations


# ---------------------------------------------------------------------------
# STEP 6 — Generate trading recommendation
# ---------------------------------------------------------------------------

def _generate_recommendation(
    yt_chunks: list[dict],
) -> Optional[dict]:
    """
    If 3+ sources agree on sentiment for a ticker, generate a brief
    trading recommendation via the LLM.
    """
    logger.info("STEP 6 — Checking for consensus recommendation …")

    # Count (symbol, sentiment) pairs across all YouTube/sentiment chunks
    symbol_sentiments: dict[str, list[str]] = {}
    for chunk in yt_chunks:
        meta = chunk.get("metadata", {})
        sym = meta.get("symbol", "")
        sent = meta.get("sentiment", "")
        if sym and sym != "N/A" and sent and sent != "N/A":
            symbol_sentiments.setdefault(sym, []).append(sent)

    # Find a ticker with 3+ agreeing sources
    for symbol, sentiments in symbol_sentiments.items():
        counter = Counter(sentiments)
        most_common_sent, count = counter.most_common(1)[0]

        if count >= _MIN_SOURCES_FOR_REC:
            logger.info(
                "  Consensus found: %s → %s (%d sources)",
                symbol, most_common_sent, count,
            )

            prompt = _RECOMMENDATION_PROMPT_TEMPLATE.format(symbol=symbol)
            summary = (
                f"{count} out of {len(sentiments)} sources are "
                f"{most_common_sent} on {symbol}."
            )

            try:
                reasoning = _call_llm_with_retry([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": summary},
                ], temperature=0.2, max_tokens=512)
            except Exception as exc:
                logger.warning("  Recommendation LLM call failed: %s", exc)
                reasoning = summary

            # Map sentiment to direction
            direction_map = {
                "bullish": "buy",
                "bearish": "sell",
                "neutral": "hold",
            }
            direction = direction_map.get(most_common_sent, "hold")

            # Confidence based on source count
            if count >= 5:
                confidence = "high"
            elif count >= 3:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "disclaimer": "Not financial advice. Always do your own research.",
            }

    logger.info("  No consensus found (need %d+ agreeing sources)", _MIN_SOURCES_FOR_REC)
    return None


# ---------------------------------------------------------------------------
# Main entry point — answer_query
# ---------------------------------------------------------------------------

from agents.hermes_utils import tool

@tool(
    name="answer_market_query",
    description="End-to-end RAG pipeline: extract tickers, retrieve context, generate a grounded answer with citations, and optionally a trading recommendation.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language user question about Indian markets."
            }
        },
        "required": ["query"]
    },
    toolset="dalal_radar"
)
def answer_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    End-to-end RAG pipeline: extract tickers, retrieve context, generate
    a grounded answer with citations, and optionally a trading
    recommendation.

    Parameters
    ----------
    query : str
        Natural-language user question about Indian markets.
    conversation_history : list[dict], optional
        Previous turns as ``[{"role": "user"/"assistant", "content": …}]``
        for multi-turn support.

    Returns
    -------
    dict
        ``{answer, citations, recommendation, sources_used, query}``
    """
    if conversation_history is None:
        conversation_history = []

    logger.info("=" * 60)
    logger.info("  Chat Agent — Processing query: %s", query[:80])
    logger.info("=" * 60)

    # STEP 1 — Extract tickers
    tickers = _extract_tickers(query)

    # STEP 2 — Retrieve context
    results = _retrieve_context(query, tickers)

    # STEP 3 — Build context string
    context_str, yt_chunks, mkt_chunks = _build_context(results)

    if not context_str.strip():
        logger.warning("  No context found — returning empty response")
        return {
            "answer": (
                "No data available for this query in the last 24 hours. "
                "Please ensure the data pipeline has been run recently."
            ),
            "citations": [],
            "recommendation": None,
            "sources_used": 0,
            "query": query,
        }

    # STEP 4 — Generate answer
    answer = _generate_answer(query, context_str, conversation_history)

    # STEP 5 — Extract citations
    citations = _extract_citations(answer, yt_chunks, mkt_chunks)

    # STEP 6 — Generate recommendation (only if consensus exists)
    recommendation = _generate_recommendation(yt_chunks)

    total_sources = len(yt_chunks) + len(mkt_chunks)
    logger.info("=" * 60)
    logger.info(
        "  Chat Agent — Done | %d sources, %d citations, rec=%s",
        total_sources, len(citations),
        recommendation["symbol"] if recommendation else "None",
    )
    logger.info("=" * 60)

    return {
        "answer": answer,
        "citations": citations,
        "recommendation": recommendation,
        "sources_used": total_sources,
        "query": query,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def format_response(response: dict) -> str:
    """
    Format an ``answer_query()`` response dict into a readable CLI string.

    Parameters
    ----------
    response : dict
        Output from ``answer_query()``.

    Returns
    -------
    str
        Multi-line formatted string.
    """
    lines: list[str] = []
    lines.append("┌" + "─" * 68 + "┐")
    lines.append(f"│  Query: {response.get('query', '')[:58]:<58} │")
    lines.append("├" + "─" * 68 + "┤")
    lines.append("│  ANSWER                                                            │")
    lines.append("├" + "─" * 68 + "┤")

    # Word-wrap the answer to fit the box
    answer = response.get("answer", "")
    for para in answer.split("\n"):
        while len(para) > 66:
            lines.append(f"│ {para[:66]} │")
            para = para[66:]
        lines.append(f"│ {para:<66} │")

    lines.append("├" + "─" * 68 + "┤")

    # Citations
    citations = response.get("citations", [])
    lines.append(f"│  CITATIONS ({len(citations)})                                                  │")
    lines.append("├" + "─" * 68 + "┤")
    if citations:
        for c in citations:
            cit_line = (
                f"{c.get('channel', '?')} — {c.get('symbol', '?')} "
                f"({c.get('sentiment', '?')})"
            )
            lines.append(f"│   • {cit_line:<63} │")
    else:
        lines.append(f"│   {'(none)':<65} │")

    # Recommendation
    rec = response.get("recommendation")
    lines.append("├" + "─" * 68 + "┤")
    lines.append(f"│  RECOMMENDATION                                                    │")
    lines.append("├" + "─" * 68 + "┤")
    if rec:
        rec_summary = (
            f"{rec['symbol']} → {rec['direction'].upper()} "
            f"(confidence: {rec['confidence']})"
        )
        lines.append(f"│   {rec_summary:<65} │")
        # Wrap reasoning
        reasoning = rec.get("reasoning", "")
        for para in reasoning.split("\n"):
            while len(para) > 64:
                lines.append(f"│   {para[:64]} │")
                para = para[64:]
            lines.append(f"│   {para:<65} │")
        lines.append(f"│   ⚠ {rec['disclaimer']:<63} │")
    else:
        lines.append(f"│   {'No recommendation (insufficient consensus)':<65} │")

    lines.append("├" + "─" * 68 + "┤")
    lines.append(f"│  Sources used: {response.get('sources_used', 0):<53} │")
    lines.append("└" + "─" * 68 + "┘")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live test — __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Chat Agent Live Test")
    print("=" * 70)

    test_queries = [
        "What is the sentiment on RELIANCE?",
        "Why are people bullish on HDFC Bank?",
        "What are FIIs doing in the market?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"  TEST {i}: {q}")
        print(f"{'─' * 70}\n")

        result = answer_query(q)
        print(format_response(result))

    print("\n" + "=" * 70)
