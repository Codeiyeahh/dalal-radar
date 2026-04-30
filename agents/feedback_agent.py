"""
Feedback Agent Module.

Implements a closed learning loop: logs every chat interaction,
accepts user ratings, and triggers an LLM rewrite when the user
flags a poor response.

Persistent store: ``./data/feedback_store.json``
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional
import time


import httpx

# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, FALLBACK_MODELS, logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)
_STORE_PATH = os.path.join(_DATA_DIR, "feedback_store.json")

_OPENROUTER_URL = OPENROUTER_BASE_URL
_LLM_MODEL = OPENROUTER_MODEL

_IMPROVE_SYSTEM_PROMPT = (
    "The user rated this response poorly. "
    "Rewrite the response to be more accurate and helpful. "
    "Cite sources properly. Do not hallucinate."
)


# ---------------------------------------------------------------------------
# JSON store helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    """Create the data directory if it doesn't exist."""
    os.makedirs(_DATA_DIR, exist_ok=True)


def _load_store() -> dict:
    """Load the feedback store from disk, or return an empty skeleton."""
    _ensure_data_dir()
    if os.path.isfile(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted feedback store — resetting (%s)", exc)
    return {"sessions": []}


def _save_store(store: dict) -> None:
    """Persist the feedback store to disk."""
    _ensure_data_dir()
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def _find_session(store: dict, session_id: str) -> Optional[dict]:
    """Return the session dict with the given ID, or None."""
    for s in store["sessions"]:
        if s["session_id"] == session_id:
            return s
    return None


# ---------------------------------------------------------------------------
# LLM Logic
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_interaction(
    session_id: str,
    query: str,
    response: dict,
    retrieved_chunks: list,
) -> str:
    """
    Save a chat interaction to the feedback store.

    Parameters
    ----------
    session_id : str
        Caller-supplied session ID (or empty to auto-generate).
    query : str
        The user's original query.
    response : dict
        Full response dict from ``chat_agent.answer_query()``.
    retrieved_chunks : list
        The raw chunks that were used as context.

    Returns
    -------
    str
        The session ID (generated if not supplied).
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:12]

    store = _load_store()

    session = {
        "session_id": session_id,
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "rating": None,
        "user_comment": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "improved": False,
    }

    store["sessions"].append(session)
    _save_store(store)

    logger.info("log_interaction -> session %s saved", session_id)
    return session_id


def submit_feedback(
    session_id: str,
    rating: int,
    comment: str = "",
) -> dict:
    """
    Attach a user rating (1-5) and optional comment to a session.

    If the rating is 1 or 2, ``improve_response()`` is called
    automatically to generate a better answer.

    Parameters
    ----------
    session_id : str
        The session to rate.
    rating : int
        Score from 1 (poor) to 5 (excellent).
    comment : str, optional
        Free-form user feedback.

    Returns
    -------
    dict
        ``{status, session_id, improved_response?}``
    """
    rating = max(1, min(5, int(rating)))

    store = _load_store()
    session = _find_session(store, session_id)

    if session is None:
        logger.warning("submit_feedback: session %s not found", session_id)
        return {"status": "error", "message": f"Session {session_id} not found"}

    session["rating"] = rating
    session["user_comment"] = comment
    _save_store(store)

    logger.info(
        "submit_feedback -> session %s rated %d/5", session_id, rating
    )

    result: dict = {"status": "ok", "session_id": session_id}

    # Trigger improvement for poor ratings
    if rating <= 2:
        improved = improve_response(session)
        # Re-load and persist the improved session
        store = _load_store()
        target = _find_session(store, session_id)
        if target is not None:
            target["improved"] = True
            target["improved_response"] = improved
            _save_store(store)
        result["improved_response"] = improved

    return result


def improve_response(session: dict) -> str:
    """
    Re-generate a better answer when the user rates poorly.

    Parameters
    ----------
    session : dict
        The full session record from the feedback store.

    Returns
    -------
    str
        The improved response text.
    """
    logger.info("improve_response -> rewriting for session %s", session["session_id"])

    query = session.get("query", "")
    original = session.get("response", {}).get("answer", "")
    comment = session.get("user_comment", "")
    chunks = session.get("retrieved_chunks", [])

    # Build a concise context summary from retrieved chunks
    context_lines: list[str] = []
    for c in chunks[:10]:  # cap to avoid token overflow
        if isinstance(c, dict):
            context_lines.append(c.get("text", str(c))[:300])
        else:
            context_lines.append(str(c)[:300])
    context_str = "\n".join(context_lines)

    user_msg = (
        f"Query: {query}\n"
        f"Original response: {original}\n"
        f"User comment: {comment}\n"
        f"Retrieved context:\n{context_str}\n\n"
        f"Rewrite the response to be more accurate and helpful."
    )

    try:
        messages = [
            {"role": "system", "content": _IMPROVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        improved = _call_llm(messages=messages)
    except Exception as exc:
        logger.error("improve_response LLM call failed: %s", exc)
        improved = "(Improvement failed — please try again later.)"

    logger.info("improve_response -> done (%d chars)", len(improved))
    return improved


def get_feedback_stats() -> dict:
    """
    Compute aggregate statistics across all feedback sessions.

    Returns
    -------
    dict
        ``{total_sessions, avg_rating, low_rated_count,
        improved_count, top_queries}``
    """
    store = _load_store()
    sessions = store.get("sessions", [])

    total = len(sessions)
    ratings = [s["rating"] for s in sessions if s.get("rating") is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
    low_rated = sum(1 for r in ratings if r <= 2)
    improved = sum(1 for s in sessions if s.get("improved"))

    # Top queries by frequency
    from collections import Counter
    query_counts = Counter(s.get("query", "") for s in sessions)
    top_queries = [
        {"query": q, "count": c}
        for q, c in query_counts.most_common(10)
    ]

    stats = {
        "total_sessions": total,
        "avg_rating": avg_rating,
        "low_rated_count": low_rated,
        "improved_count": improved,
        "top_queries": top_queries,
    }

    logger.info("get_feedback_stats -> %s", stats)
    return stats


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Feedback Agent Smoke Test")
    print("=" * 70)

    # Log a fake interaction
    fake_response = {
        "answer": "Reliance is bullish according to 3 creators.",
        "citations": [{"channel": "TestTrader", "symbol": "RELIANCE"}],
        "recommendation": None,
        "sources_used": 3,
        "query": "What is the sentiment on RELIANCE?",
    }
    fake_chunks = [{"text": "Reliance bullish on daily chart", "symbol": "RELIANCE"}]

    print("\n[1] Logging interaction …")
    sid = log_interaction("", "What is the sentiment on RELIANCE?", fake_response, fake_chunks)
    print(f"    Session ID: {sid}")

    print("\n[2] Submitting positive feedback (4/5) …")
    res = submit_feedback(sid, 4, "Good answer!")
    print(f"    Result: {res}")

    print("\n[3] Logging another interaction …")
    sid2 = log_interaction("", "Why is Infosys falling?", fake_response, fake_chunks)

    print("\n[4] Submitting negative feedback (1/5) — triggers improvement …")
    res2 = submit_feedback(sid2, 1, "The answer was wrong and vague.")
    print(f"    Result status: {res2.get('status')}")
    if res2.get("improved_response"):
        print(f"    Improved response: {res2['improved_response'][:120]}…")

    print("\n[5] Feedback stats:")
    stats = get_feedback_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 70)
