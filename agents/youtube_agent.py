"""
YouTube Agent Module.

Scrapes ~100 Indian trading YouTube creators, fetches their recent video
transcripts, and runs LLM-powered ticker sentiment analysis on them.

Pipeline
--------
1. fetch_recent_videos()        — Apify "streamers/youtube-scraper"
2. fetch_transcripts(videos)    — Apify "apify/youtube-transcript-scraper"
3. extract_ticker_sentiment()   — OpenRouter (Mistral-7B-Instruct)
4. run_youtube_pipeline()       — Orchestrator with retry logic
5. format_for_rag(results)      — Convert to RAG-ready text chunks
"""

from __future__ import annotations

import json
import re
import sys
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from apify_client import ApifyClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig


# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APIFY_TOKEN, OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEARCH_QUERIES: list[str] = [
    "NSE trading",
    "Indian stock market today",
    "Nifty analysis",
    "Bank Nifty tips",
    "stock market hindi",
    "trading strategies NSE",
]

# Target total videos across all queries
_TARGET_TOTAL_VIDEOS = 100
_VIDEOS_PER_QUERY = _TARGET_TOTAL_VIDEOS // len(SEARCH_QUERIES)  # ~16 each

# Apify actor IDs
_YOUTUBE_SCRAPER_ACTOR = "streamers/youtube-scraper"
_TRANSCRIPT_SCRAPER_ACTOR = "tri_angle/youtube-transcript-scraper"


# OpenRouter LLM settings
_OPENROUTER_URL = OPENROUTER_BASE_URL
_LLM_MODEL = OPENROUTER_MODEL
_LLM_SYSTEM_PROMPT = (
    "You are a financial analyst. Extract all NSE/BSE stock tickers "
    "mentioned in this transcript. For each ticker, determine if the "
    "creator is bullish, bearish or neutral. Return JSON only:\n"
    '{tickers: [{symbol, sentiment, reason, confidence}]}'
)

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry_with_backoff(func, *args, max_retries: int = _MAX_RETRIES, **kwargs):
    """
    Execute *func* with exponential backoff retry.

    Parameters
    ----------
    func : callable
        The function to call.
    max_retries : int
        Maximum number of retry attempts (default 3).

    Returns
    -------
    Any
        Whatever *func* returns on a successful call.

    Raises
    ------
    Exception
        The last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = _BACKOFF_BASE_SECONDS ** attempt
                logger.warning(
                    "Attempt %d/%d for %s failed (%s) — retrying in %ds",
                    attempt, max_retries, func.__name__, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "All %d retries exhausted for %s. Last error: %s",
                    max_retries, func.__name__, exc,
                )
    raise last_exc  # type: ignore[misc]


def _is_within_24h(published_at: str) -> bool:
    """Return True if the ISO-8601 timestamp is within the last 24 hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        # Apify usually returns ISO 8601 with 'Z' suffix
        dt_str = published_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= cutoff
    except (ValueError, AttributeError):
        logger.debug("Could not parse date '%s', including video anyway", published_at)
        return True


def _get_apify_client() -> ApifyClient:
    """Build and return an authenticated Apify client."""
    if not APIFY_TOKEN:
        raise RuntimeError(
            "APIFY_TOKEN is not set. Add it to your .env file."
        )
    return ApifyClient(APIFY_TOKEN)


# ---------------------------------------------------------------------------
# STEP 1 — Fetch recent videos
# ---------------------------------------------------------------------------

def fetch_recent_videos() -> list[dict]:
    """
    Search YouTube for recent Indian trading / stock-market videos.

    Uses the Apify ``streamers/youtube-scraper`` actor with each query
    from SEARCH_QUERIES.  Only videos published in the last 24 hours
    are kept.

    Returns
    -------
    list[dict]
        Each dict has keys: video_id, title, channel_name,
        published_at, url.
    """
    logger.info("STEP 1 — Fetching recent YouTube videos …")
    client = _get_apify_client()
    all_videos: list[dict] = []
    seen_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        logger.info("  Searching: '%s' (target %d per query)", query, _VIDEOS_PER_QUERY)

        run_input = {
            "searchKeywords": query,
            "maxResults": _VIDEOS_PER_QUERY + 5,  # slight over-fetch to offset dedup
            "maxResultsShorts": 0,
            "startUrls": [],
            "type": "search",
        }

        try:
            run = client.actor(_YOUTUBE_SCRAPER_ACTOR).call(run_input=run_input)
            dataset_items = list(
                client.dataset(run["defaultDatasetId"]).iterate_items()
            )
        except Exception as exc:
            logger.warning("  Apify search for '%s' failed: %s", query, exc)
            continue

        count_added = 0
        for item in dataset_items:
            vid_id = item.get("id") or item.get("videoId") or ""
            if not vid_id or vid_id in seen_ids:
                continue

            published = item.get("date") or item.get("uploadDate") or ""
            if not _is_within_24h(published):
                continue

            seen_ids.add(vid_id)
            all_videos.append({
                "video_id": vid_id,
                "title": item.get("title", "N/A"),
                "channel_name": item.get("channelName") or item.get("channelTitle", "N/A"),
                "published_at": published,
                "url": item.get("url") or f"https://www.youtube.com/watch?v={vid_id}",
            })
            count_added += 1

        logger.info("  -> Added %d videos from '%s'", count_added, query)

    logger.info("STEP 1 complete — %d unique videos in last 24h", len(all_videos))
    return all_videos


# ---------------------------------------------------------------------------
# STEP 2 — Fetch transcripts
# ---------------------------------------------------------------------------

def fetch_transcripts(videos: list[dict]) -> dict[str, str]:
    """
    Fetch transcripts for the given videos via Apify, with a local fallback.

    Parameters
    ----------
    videos : list[dict]
        Output from ``fetch_recent_videos()``.

    Returns
    -------
    dict[str, str]
        Mapping of video_id → transcript text.
    """
    logger.info("STEP 2 — Fetching transcripts for %d videos …", len(videos))
    if not videos:
        return {}

    transcripts: dict[str, str] = {}
    
    # --- Try Apify First ---
    try:
        client = _get_apify_client()
        run_input = {
            "urls": [v["url"] for v in videos],
            "language": "en"
        }
        
        logger.info("  -> Calling Apify actor: %s for %d videos...", _TRANSCRIPT_SCRAPER_ACTOR, len(videos))
        run = client.actor(_TRANSCRIPT_SCRAPER_ACTOR).call(run_input=run_input)
        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        # Build a quick URL -> video_id lookup
        url_to_id: dict[str, str] = {}
        for v in videos:
            url_to_id[v["url"]] = v["video_id"]
            url_to_id[v["video_id"]] = v["video_id"]

        for item in dataset_items:
            source_url = item.get("url") or item.get("inputUrl") or ""
            vid_id = item.get("videoId") or url_to_id.get(source_url, "")
            
            if not vid_id and source_url:
                for v in videos:
                    if v["video_id"] in source_url:
                        vid_id = v["video_id"]
                        break
            
            if not vid_id:
                continue

            transcript_text = ""
            captions = item.get("captions") or item.get("transcript") or []
            if isinstance(captions, list):
                transcript_text = " ".join(seg.get("text", "") for seg in captions if isinstance(seg, dict))
            elif isinstance(captions, str):
                transcript_text = captions
            
            if not transcript_text:
                transcript_text = item.get("text", "")

            if transcript_text.strip():
                transcripts[vid_id] = transcript_text.strip()
                
        if transcripts:
            logger.info("  -> Successfully fetched %d transcripts via Apify", len(transcripts))
    except Exception as exc:
        logger.warning("  Apify transcript actor failed or not found: %s. Falling back to youtube-transcript-api...", exc)

    # --- Local Fallback for Missing Transcripts ---
    missing_ids = [v["video_id"] for v in videos if v["video_id"] not in transcripts]
    if missing_ids:
        logger.info("  Attempting local fallback for %d missing transcript(s)...", len(missing_ids))
        
        # Configure Proxies
        webshare_user = os.getenv("WEBSHARE_USERNAME", "")
        webshare_pass = os.getenv("WEBSHARE_PASSWORD", "")
        
        if webshare_user and webshare_pass:
            logger.info("  Using Webshare proxies for local fallback...")
            proxy_config = WebshareProxyConfig(
                proxy_username=webshare_user,
                proxy_password=webshare_pass
            )
            api = YouTubeTranscriptApi(proxies=proxy_config)
        else:
            logger.warning("  WEBSHARE credentials not found. Using direct connection (may be blocked).")
            api = YouTubeTranscriptApi()

        for vid_id in missing_ids:
            try:
                # Local library fallback with language support (English and Hindi)
                srt_obj = api.fetch(vid_id, languages=['en', 'hi', 'en-IN'])
                srt = srt_obj.to_raw_data()
                text = " ".join([entry['text'] for entry in srt])
                if text.strip():
                    transcripts[vid_id] = text.strip()
                    logger.info(f"Fallback success: {vid_id}")
            except Exception as e:
                logger.warning(f"Fallback failed for {vid_id}: {type(e).__name__}: {e}")



    logger.info(
        "STEP 2 complete — transcripts fetched: %d / %d",
        len(transcripts), len(videos),
    )
    return transcripts



# ---------------------------------------------------------------------------
# STEP 3 — LLM-powered ticker sentiment extraction
# ---------------------------------------------------------------------------

def _parse_llm_json(raw_text: str) -> list[dict]:
    """
    Best-effort extraction of the JSON ticker list from the LLM response.

    The model sometimes wraps JSON in markdown code fences or adds prose.
    """
    # Try to find JSON object with a 'tickers' key
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`")

    # Try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "tickers" in parsed:
            return parsed["tickers"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object within the text
    match = re.search(r'\{[^{}]*"tickers"\s*:\s*\[.*?\]\s*\}', cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed.get("tickers", [])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM JSON response — returning empty list")
    return []


def extract_ticker_sentiment(transcript: str, video_meta: dict) -> list[dict]:
    """
    Use OpenRouter (Mistral-7B-Instruct) to extract stock ticker
    sentiments from a video transcript.

    Parameters
    ----------
    transcript : str
        The full transcript text.
    video_meta : dict
        Video metadata dict (from Step 1).

    Returns
    -------
    list[dict]
        Each dict: {symbol, sentiment, reason, confidence,
        channel_name, video_id, timestamp, video_url}
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    # Rate limit handling: 8 second sleep between calls
    time.sleep(8)


    # Truncate very long transcripts to avoid token limits
    max_chars = 12_000
    truncated = transcript[:max_chars]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dalal-radar",
        "X-Title": "Dalal Radar",
    }

    payload = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyze this Indian stock market YouTube video transcript.\n"
                    f"Channel: {video_meta.get('channel_name', 'Unknown')}\n"
                    f"Title: {video_meta.get('title', 'Unknown')}\n\n"
                    f"TRANSCRIPT:\n{truncated}"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    with httpx.Client(timeout=60) as http_client:
        resp = http_client.post(_OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()

    data = resp.json()
    raw_content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    tickers = _parse_llm_json(raw_content)

    # Enrich each ticker with video metadata
    enriched: list[dict] = []
    for t in tickers:
        enriched.append({
            "symbol": str(t.get("symbol", "")).strip().upper(),
            "sentiment": str(t.get("sentiment", "neutral")).strip().lower(),
            "reason": str(t.get("reason", "")),
            "confidence": t.get("confidence", 0),
            "channel_name": video_meta.get("channel_name", "N/A"),
            "video_id": video_meta.get("video_id", ""),
            "timestamp": video_meta.get("published_at", ""),
            "video_url": video_meta.get("url", ""),
        })

    return enriched


# ---------------------------------------------------------------------------
# STEP 4 — Orchestrator pipeline
# ---------------------------------------------------------------------------
from agents.hermes_utils import tool

@tool(
    name="run_youtube_pipeline",
    description="Fetch recent YouTube videos and extract stock sentiment using an LLM.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    toolset="youtube"
)
def run_youtube_pipeline() -> list[dict]:
    """
    Orchestrates the full YouTube sentiment pipeline:

    1. Fetch recent videos  (with retry)
    2. Fetch transcripts    (with retry)
    3. Run sentiment extraction on each transcript (with per-video retry)

    Returns
    -------
    list[dict]
        Flat list of all ticker-sentiment results across all videos.
    """
    logger.info("=" * 60)
    logger.info("  YouTube Pipeline — Starting")
    logger.info("=" * 60)

    # Step 1 — fetch videos
    videos = _retry_with_backoff(fetch_recent_videos)
    if not videos:
        logger.warning("No videos found — aborting pipeline.")
        return []

    # Step 2 — fetch transcripts
    transcripts = _retry_with_backoff(fetch_transcripts, videos)
    if not transcripts:
        logger.warning("No transcripts fetched — aborting pipeline.")
        return []

    # Step 3 — extract sentiment from each transcript
    all_sentiments: list[dict] = []
    total = len(transcripts)
    logger.info("STEP 3 — Running sentiment extraction on %d transcripts …", total)

    # Build video_id -> meta lookup for fast access
    meta_lookup: dict[str, dict] = {v["video_id"]: v for v in videos}

    for idx, (vid_id, transcript) in enumerate(transcripts.items(), start=1):
        # Rate limit handling: Process in batches of 5
        if idx > 1 and (idx - 1) % 5 == 0:
            logger.info("Rate limit pause... (10s between batches)")
            time.sleep(10)

        meta = meta_lookup.get(vid_id, {"video_id": vid_id})
        logger.info(
            "  [%d/%d] Analyzing: %s — %s",
            idx, total,
            meta.get("channel_name", "?"),
            meta.get("title", "?")[:60],
        )

        try:
            results = _retry_with_backoff(
                extract_ticker_sentiment, transcript, meta
            )
            all_sentiments.extend(results)
            logger.info("    -> %d ticker(s) extracted", len(results))
        except Exception as exc:
            logger.error(
                "    -> Sentiment extraction failed for %s: %s", vid_id, exc
            )


    logger.info("=" * 60)
    logger.info(
        "  YouTube Pipeline — Complete | %d tickers from %d videos",
        len(all_sentiments), total,
    )
    logger.info("=" * 60)

    return all_sentiments


# ---------------------------------------------------------------------------
# STEP 5 — Format for RAG
# ---------------------------------------------------------------------------

def format_for_rag(results: list[dict]) -> list[str]:
    """
    Convert sentiment results into readable text chunks for RAG injection.

    Each chunk is a single-line summary of one ticker mention.

    Parameters
    ----------
    results : list[dict]
        Output from ``run_youtube_pipeline()``.

    Returns
    -------
    list[str]
        List of human-readable strings (one per ticker mention).
    """
    chunks: list[str] = []
    for r in results:
        chunk = (
            f"Channel: {r.get('channel_name', 'N/A')} | "
            f"Video: {r.get('video_id', 'N/A')} | "
            f"Ticker: {r.get('symbol', 'N/A')} | "
            f"Sentiment: {r.get('sentiment', 'neutral')} | "
            f"Reason: {r.get('reason', 'N/A')} | "
            f"URL: {r.get('video_url', 'N/A')} | "
            f"Time: {r.get('timestamp', 'N/A')}"
        )
        chunks.append(chunk)

    logger.info("format_for_rag -> %d RAG chunks generated", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Live test — __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — YouTube Agent Live Test")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # [0] Quick Library Smoke Test
    # ---------------------------------------------------------------------------
    print("\n[0] Testing youtube-transcript-api directly (dVd9kdTTLLo) …")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.proxies import WebshareProxyConfig
        
        web_u = os.getenv("WEBSHARE_USERNAME", "")
        web_p = os.getenv("WEBSHARE_PASSWORD", "")
        
        if web_u and web_p:
            print("    Configuring Webshare proxies for test...")
            p_config = WebshareProxyConfig(proxy_username=web_u, proxy_password=web_p)
            ytt_api = YouTubeTranscriptApi(proxies=p_config)
        else:
            print("    No Webshare credentials. Testing direct connection...")
            ytt_api = YouTubeTranscriptApi()
            
        test_transcript_obj = ytt_api.fetch('dVd9kdTTLLo', languages=['en', 'hi', 'en-IN'])
        test_transcript = test_transcript_obj.to_raw_data()
        print(f"    Success! First 3 lines: {test_transcript[:3]}")
    except Exception as exc:
        if "IpBlocked" in str(exc):
            print("    [!] Local IP is blocked by YouTube. This is common on cloud/server IPs.")
            print("        Pipeline will rely on Apify proxies instead.")
        else:
            print(f"    Direct test failed: {type(exc).__name__}: {exc}")


    print("\n[1] Running full YouTube pipeline …\n")

    sentiments = run_youtube_pipeline()

    if sentiments:
        print(f"\n[2] Total ticker mentions extracted: {len(sentiments)}\n")

        # Show first 10 results as a sample
        sample = sentiments[:10]
        print(f"    Showing first {len(sample)} result(s):\n")
        for i, s in enumerate(sample, 1):
            print(
                f"    {i:>3}. {s['symbol']:<12} | "
                f"{s['sentiment']:<8} | "
                f"Channel: {s['channel_name']:<25} | "
                f"Reason: {s['reason'][:50]}"
            )

        print(f"\n[3] RAG chunks (first 5):\n")
        rag_chunks = format_for_rag(sentiments)
        for chunk in rag_chunks[:5]:
            print(f"    {chunk}")
    else:
        print("    No sentiment data returned.")
        print("    Check your APIFY_TOKEN and OPENROUTER_API_KEY in .env")

    print("\n" + "=" * 70)
