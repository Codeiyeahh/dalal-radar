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
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from apify_client import ApifyClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APIFY_TOKEN, OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, FALLBACK_MODELS, logger  # noqa: E402

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
    """
    Return True if the timestamp or relative time string is within 24h.
    Handles ISO-8601 ('2026-04-29...') and relative ('5 hours ago').
    """
    if not published_at:
        return True
    
    # Handle relative time strings (from direct scraper)
    low = published_at.lower()
    if any(kw in low for kw in ["hour", "minute", "second", "1 day", "just now"]):
        # If it says "days" (plural) and not "1 day", it's > 24h
        if "day" in low and "1 day" not in low:
            return False
        return True
    if "day" in low: # e.g. "2 days ago"
        return False
    if "week" in low or "month" in low or "year" in low:
        return False

    # Handle ISO-8601 (from Apify)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        dt_str = published_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= cutoff
    except (ValueError, AttributeError):
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

def _fetch_direct_search(query: str, max_results: int) -> list[dict]:
    """
    Directly scrapes YouTube search results using urllib.
    No API key required. Works best from a non-server (residential) IP.
    """
    import urllib.request
    import urllib.parse

    encoded_query = urllib.parse.quote(query)
    # sp=CAI%253D sorts by upload date
    url = f"https://www.youtube.com/results?search_query={encoded_query}&sp=CAI%253D"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')
            
            match = re.search(r'var ytInitialData = (\{.*?\});', html)
            if not match:
                match = re.search(r'>window\["ytInitialData"\] = (\{.*?\});<', html)
            if not match:
                return []
                
            data = json.loads(match.group(1))
            videos = []
            
            try:
                # Path to video listings in the JSON blob
                contents = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
                for section in contents:
                    if "itemSectionRenderer" in section:
                        items = section["itemSectionRenderer"]["contents"]
                        for item in items:
                            if "videoRenderer" in item:
                                vr = item["videoRenderer"]
                                vid_id = vr["videoId"]
                                title_text = vr["title"]["runs"][0]["text"]
                                channel_name = vr["shortBylineText"]["runs"][0]["text"]
                                
                                # Basic 24h filter: only if it contains 'hour', 'minute', or '1 day'
                                pub_text = vr.get("publishedTimeText", {}).get("simpleText", "")
                                if not any(kw in pub_text.lower() for kw in ["hour", "minute", "second", "1 day", "just now"]):
                                    # If it says "2 days ago", skip
                                    if "day" in pub_text.lower() and "1" not in pub_text:
                                        continue
                                    # If it has no time text, include anyway (might be new)
                                    if pub_text: continue

                                # Extract description snippet
                                desc_snippet = ""
                                if "descriptionSnippet" in vr:
                                    runs = vr["descriptionSnippet"].get("runs", [])
                                    desc_snippet = "".join(r.get("text", "") for r in runs)

                                videos.append({
                                    "video_id": vid_id,
                                    "title": title_text,
                                    "channel_name": channel_name,
                                    "description": desc_snippet,
                                    "published_at": pub_text or "Recently",
                                    "url": f"https://www.youtube.com/watch?v={vid_id}"
                                })
                                if len(videos) >= max_results:
                                    break
            except Exception as e:
                logger.debug("Error parsing search results for '%s': %s", query, e)
            return videos
    except Exception as e:
        logger.warning("Direct search failed for '%s': %s", query, e)
        return []




def fetch_recent_videos() -> list[dict]:
    """
    Search YouTube for recent Indian trading / stock-market videos.
    Uses direct scraping as primary discovery (no API key required).
    """
    logger.info("STEP 1 — Fetching recent YouTube videos …")
    all_videos: list[dict] = []
    seen_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        logger.info("  Searching: '%s' (target %d per query)", query, _VIDEOS_PER_QUERY)
        
        # 1. Try Direct Scraper (Residential IP friendly)
        query_videos = _fetch_direct_search(query, _VIDEOS_PER_QUERY)
        
        # 2. Fallback to Apify ONLY if direct failed and token exists
        if not query_videos and APIFY_TOKEN:
            logger.info("    (Direct search empty/failed, trying Apify fallback...)")
            try:
                client = _get_apify_client()
                run_input = {
                    "searchKeywords": query,
                    "maxResults": _VIDEOS_PER_QUERY,
                    "maxResultsShorts": 0,
                    "type": "search",
                }
                run = client.actor(_YOUTUBE_SCRAPER_ACTOR).call(run_input=run_input)
                dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                
                for item in dataset_items:
                    vid_id = item.get("id") or item.get("videoId") or ""
                    if not vid_id: continue
                    published = item.get("date") or item.get("uploadDate") or ""
                    if not _is_within_24h(published): continue
                    
                    query_videos.append({
                        "video_id": vid_id,
                        "title": item.get("title", "N/A"),
                        "channel_name": item.get("channelName") or item.get("channelTitle", "N/A"),
                        "description": item.get("description", ""),
                        "published_at": published,
                        "url": item.get("url") or f"https://www.youtube.com/watch?v={vid_id}",
                    })
            except Exception as exc:
                logger.warning("    Apify fallback failed for '%s': %s", query, exc)

        # Dedup and add to global list
        added_this_query = 0
        for v in query_videos:
            if v["video_id"] not in seen_ids:
                seen_ids.add(v["video_id"])
                all_videos.append(v)
                added_this_query += 1
        
        logger.info("    -> Added %d videos for '%s'", added_this_query, query)

    logger.info("STEP 1 complete — %d unique videos found", len(all_videos))
    return all_videos


# ---------------------------------------------------------------------------
# STEP 2 — Fetch transcripts
# ---------------------------------------------------------------------------

def _fetch_single_transcript(v, proxy_config):
    vid_id = v["video_id"]
    # 1. Try Direct
    try:
        srt = YouTubeTranscriptApi.get_transcript(vid_id, languages=['en', 'hi', 'en-IN'])
        text = " ".join([entry['text'] for entry in srt])
        if text.strip():
            return vid_id, text.strip(), "direct"
    except Exception:
        pass
        
    # 2. Try Proxy Fallback
    if proxy_config:
        try:
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
            srt_obj = api.fetch(vid_id, languages=['en', 'hi', 'en-IN'])
            text = " ".join([entry['text'] for entry in srt_obj.to_raw_data()])
            if text.strip():
                return vid_id, text.strip(), "proxy"
        except Exception:
            pass
            
    return vid_id, None, "failed"


def fetch_transcripts(videos: list[dict]) -> dict[str, str]:
    """
    Fetch transcripts for the given videos via parallel local fetching with Apify fallback.
    """
    logger.info("STEP 2 — Fetching transcripts for %d videos …", len(videos))
    if not videos:
        return {}

    transcripts: dict[str, str] = {}
    
    # Configure Proxies for fallback
    web_u = os.getenv("WEBSHARE_USERNAME", "")
    web_p = os.getenv("WEBSHARE_PASSWORD", "")
    proxy_config = None
    if web_u and web_p:
        proxy_config = WebshareProxyConfig(proxy_username=web_u, proxy_password=web_p)

    # Use ThreadPoolExecutor for parallel transcript fetching
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_fetch_single_transcript, v, proxy_config): v 
            for v in videos
        }
        for future in as_completed(futures):
            vid_id, text, source = future.result()
            if text:
                transcripts[vid_id] = text
                logger.info("    -> %s success (%s)", vid_id, source)

    # --- C. Final Fallback to Apify for anything still missing ---
    missing_videos = [v for v in videos if v["video_id"] not in transcripts]
    if missing_videos and APIFY_TOKEN:
        logger.info("  Attempting Apify fallback for %d remaining transcripts...", len(missing_videos))
        try:
            client = _get_apify_client()
            run_input = {
                "urls": [v["url"] for v in missing_videos],
                "language": "en"
            }
            run = client.actor(_TRANSCRIPT_SCRAPER_ACTOR).call(run_input=run_input)
            dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            
            url_to_id = {v["url"]: v["video_id"] for v in missing_videos}
            for item in dataset_items:
                source_url = item.get("url") or item.get("inputUrl") or ""
                vid_id = item.get("videoId") or url_to_id.get(source_url, "")
                
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
                    logger.info("    Apify success: %s", vid_id)
        except Exception as exc:
            logger.warning("  Apify transcript fallback failed: %s", exc)

    logger.info("STEP 2 complete — transcripts fetched: %d / %d", len(transcripts), len(videos))
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


def _call_llm(system_prompt: str, user_content: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """
    Helper to call OpenRouter with dynamic prompts and fallback model support.
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
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            with httpx.Client(timeout=60) as http_client:
                resp = http_client.post(endpoint, headers=headers, json=payload)
                
                # Try fallback on any 4xx error
                if 400 <= resp.status_code < 500:
                    logger.warning("Model %s returned %d. Trying fallback...", model, resp.status_code)
                    continue
                
                resp.raise_for_status()

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                logger.info("Using model: %s", model)
                return content
            
        except Exception as e:
            last_error = e
            logger.warning("Model %s failed: %s. Trying next...", model, e)

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def extract_ticker_sentiment(transcript: str, video_meta: dict) -> list[dict]:
    """
    Use OpenRouter to extract stock ticker sentiments from a video transcript.
    """
    # Truncate very long transcripts to avoid token limits
    max_chars = 12_000
    truncated = transcript[:max_chars]

    user_content = (
        f"Analyze this Indian stock market YouTube video transcript.\n"
        f"Channel: {video_meta.get('channel_name', 'Unknown')}\n"
        f"Title: {video_meta.get('title', 'Unknown')}\n\n"
        f"TRANSCRIPT:\n{truncated}"
    )

    raw_content = _call_llm(_LLM_SYSTEM_PROMPT, user_content)


def extract_sentiment_from_description(video_meta: dict) -> list[dict]:
    """
    Extract ticker sentiment from video description and title.
    Used as fallback when transcript fetching fails.
    """
    title = video_meta.get('title', '')
    description = video_meta.get('description', '')
    channel = video_meta.get('channel_name', '')
    video_id = video_meta.get('video_id', '')
    video_url = video_meta.get('url', '')
    
    combined_text = f"Title: {title}\n\nDescription: {description}"
    
    if not title and not description:
        return []
    
    # Truncate to 3000 chars to stay within token limits
    combined_text = combined_text[:3000]
    
    prompt = """You are a financial analyst specializing in Indian markets.
    Extract all NSE/BSE stock tickers mentioned in this YouTube video 
    title and description. For each ticker determine if the creator 
    is bullish, bearish or neutral based on the language used.
    
    Return JSON only, no other text:
    {"tickers": [{"symbol": "RELIANCE", "sentiment": "bullish", 
    "reason": "brief reason from text", "confidence": "high/medium/low"}]}
    
    If no tickers found return: {"tickers": []}"""
    
    try:
        response = _call_llm(prompt, combined_text)
        parsed = _parse_llm_json(response)
        # Handle if parsed is a list or dict with 'tickers' key
        tickers = parsed if isinstance(parsed, list) else []
        
        # Enrich with video metadata
        results = []
        for t in tickers:
            results.append({
                'symbol': str(t.get('symbol', '')).strip().upper(),
                'sentiment': str(t.get('sentiment', 'neutral')).strip().lower(),
                'reason': str(t.get('reason', '')),
                'confidence': t.get('confidence', 'medium'),
                'channel_name': channel,
                'video_id': video_id,
                'video_url': video_url,
                'source_type': 'description',  # flag this clearly
                'timestamp': video_meta.get('published_at', '')
            })
        
        logger.info(f"    Description sentiment: {video_id} -> {len(results)} ticker(s) found")
        return results
        
    except Exception as e:
        logger.warning(f"    Description sentiment failed for {video_id}: {e}")
        return []




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
    Orchestrates the full YouTube sentiment pipeline using sequential analysis.
    """
    logger.info("=" * 60)
    logger.info("  YouTube Pipeline — Starting")
    logger.info("=" * 60)

    # Step 1 — fetch videos
    videos = _retry_with_backoff(fetch_recent_videos)
    if not videos:
        logger.warning("No videos found — aborting pipeline.")
        return []

    # Step 2 — fetch transcripts (Parallel)
    transcripts = _retry_with_backoff(fetch_transcripts, videos)

    # Step 3 — Sentiment Analysis (Sequential)
    all_sentiments: list[dict] = []
    from_transcripts = 0
    from_descriptions = 0
    
    total_videos = len(videos)
    logger.info("STEP 3 — Analyzing %d videos sequentially …", total_videos)

    for idx, meta in enumerate(videos, 1):
        vid_id = meta["video_id"]
        transcript = transcripts.get(vid_id)
        
        logger.info("[%d/%d] Analyzing: %s", idx, total_videos, 
                    meta.get("title", "?")[:50])
        
        if transcript:
            try:
                results = extract_ticker_sentiment(transcript, meta)
                for r in results:
                    r['source_type'] = 'transcript'
                all_sentiments.extend(results)
                from_transcripts += len(results)
            except Exception as exc:
                logger.error("Transcript analysis failed: %s", exc)
                results = extract_sentiment_from_description(meta)
                all_sentiments.extend(results)
                from_descriptions += len(results)
        else:
            results = extract_sentiment_from_description(meta)
            all_sentiments.extend(results)
            from_descriptions += len(results)
        
        # Rate limit: wait 3 seconds between each LLM call
        time.sleep(3)

    logger.info("=" * 60)
    logger.info(
        "  YouTube Pipeline — Complete | Total: %d tickers",
        len(all_sentiments)
    )
    logger.info(
        "  Breakdown: %d from transcripts, %d from descriptions",
        from_transcripts, from_descriptions
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
            f"Source: {r.get('source_type', 'unknown')} | "
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
            ytt_api = YouTubeTranscriptApi(proxy_config=p_config)
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
