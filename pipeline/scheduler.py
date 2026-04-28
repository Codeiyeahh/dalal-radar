"""
Scheduler Module.

Runs the full data-ingestion pipeline (market deals + YouTube sentiment)
on a 24-hour cycle using APScheduler, with an immediate run at startup.

Functions
---------
run_full_pipeline()   -> dict   (one-shot execution)
start_scheduler()     -> None   (background 24h loop)
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger  # noqa: E402

# Module-level scheduler reference (so we can inspect / shut down later)
_scheduler: BackgroundScheduler | None = None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline() -> dict:
    """
    Execute every step of the data-ingestion pipeline in sequence:

    1. Fetch NSE bulk/block deals        (market_agent)
    2. Format deals for RAG              (market_agent)
    3. Run YouTube sentiment pipeline    (youtube_agent)
    4. Format sentiments for RAG         (youtube_agent)
    5. Chunk market + sentiment data     (chunker)
    6. Embed all chunks                  (embedder)
    7. Persist to ChromaDB               (vector_store)

    Returns
    -------
    dict
        ``{status, chunks_stored, deals, videos_processed, duration_s}``
    """
    start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("  Scheduled Pipeline — START (%s)", start.isoformat())
    logger.info("=" * 60)

    # Lazy imports to avoid circular dependencies at module load time
    from agents.market_agent import get_all_deals, format_deals_for_rag
    from agents.youtube_agent import run_youtube_pipeline, format_for_rag
    from pipeline.chunker import chunk_market_data, chunk_sentiment_results
    from pipeline.embedder import embed_chunks
    from retrieval.vector_store import store_chunks

    all_chunks: list[dict] = []
    deal_count = 0
    video_count = 0

    # ------------------------------------------------------------------
    # STEP 1 — Market deals
    # ------------------------------------------------------------------
    logger.info("STEP 1/7 — Fetching NSE deals …")
    try:
        deals = get_all_deals()
        deal_count = len(deals)
        logger.info("  Fetched %d deal(s)", deal_count)
    except Exception as exc:
        logger.error("  Market agent failed: %s", exc)
        deals = []

    # ------------------------------------------------------------------
    # STEP 2 — Format deals for RAG
    # ------------------------------------------------------------------
    logger.info("STEP 2/7 — Formatting deals for RAG …")
    deals_text = format_deals_for_rag(deals) if deals else ""
    logger.info("  Deals text length: %d chars", len(deals_text))

    # ------------------------------------------------------------------
    # STEP 3 — YouTube pipeline
    # ------------------------------------------------------------------
    logger.info("STEP 3/7 — Running YouTube sentiment pipeline …")
    try:
        yt_results = run_youtube_pipeline()
        video_count = len(yt_results)
        logger.info("  YouTube pipeline returned %d ticker mention(s)", video_count)
    except Exception as exc:
        logger.error("  YouTube pipeline failed: %s", exc)
        yt_results = []

    # ------------------------------------------------------------------
    # STEP 4 — Format YouTube results for RAG
    # ------------------------------------------------------------------
    logger.info("STEP 4/7 — Formatting YouTube results for RAG …")
    _rag_strings = format_for_rag(yt_results) if yt_results else []
    logger.info("  RAG strings: %d", len(_rag_strings))

    # ------------------------------------------------------------------
    # STEP 5 — Chunk everything
    # ------------------------------------------------------------------
    logger.info("STEP 5/7 — Chunking data …")

    if deals_text:
        market_chunks = chunk_market_data(deals_text)
        all_chunks.extend(market_chunks)
        logger.info("  Market chunks: %d", len(market_chunks))

    if yt_results:
        sentiment_chunks = chunk_sentiment_results(yt_results)
        all_chunks.extend(sentiment_chunks)
        logger.info("  Sentiment chunks: %d", len(sentiment_chunks))

    logger.info("  Total chunks before embedding: %d", len(all_chunks))

    # ------------------------------------------------------------------
    # STEP 6 — Embed
    # ------------------------------------------------------------------
    if all_chunks:
        logger.info("STEP 6/7 — Embedding %d chunk(s) …", len(all_chunks))
        try:
            all_chunks = embed_chunks(all_chunks)
        except Exception as exc:
            logger.error("  Embedding failed: %s", exc)
            all_chunks = []
    else:
        logger.info("STEP 6/7 — No chunks to embed (skipping)")

    # ------------------------------------------------------------------
    # STEP 7 — Persist to ChromaDB
    # ------------------------------------------------------------------
    if all_chunks:
        logger.info("STEP 7/7 — Storing %d chunk(s) in ChromaDB …", len(all_chunks))
        try:
            store_chunks(all_chunks)
        except Exception as exc:
            logger.error("  ChromaDB store failed: %s", exc)
    else:
        logger.info("STEP 7/7 — No chunks to store (skipping)")

    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds()

    summary = {
        "status": "ok",
        "chunks_stored": len(all_chunks),
        "deals": deal_count,
        "videos_processed": video_count,
        "duration_s": round(duration, 2),
    }

    logger.info("=" * 60)
    logger.info("  Scheduled Pipeline — DONE in %.1fs", duration)
    logger.info("  Summary: %s", summary)
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------

def start_scheduler() -> None:
    """
    Start an APScheduler ``BackgroundScheduler`` that runs
    ``run_full_pipeline()`` every 24 hours.

    The pipeline is also triggered immediately on the first call.
    """
    global _scheduler

    if _scheduler is not None:
        logger.warning("Scheduler already running — skipping duplicate start")
        return

    logger.info("Initialising APScheduler (interval=24h) …")
    _scheduler = BackgroundScheduler(daemon=True)

    _scheduler.add_job(
        run_full_pipeline,
        trigger="interval",
        hours=24,
        id="dalal_radar_pipeline",
        name="Dalal Radar 24h Pipeline",
        replace_existing=True,
    )

    _scheduler.start()

    jobs = _scheduler.get_jobs()
    if jobs:
        next_run = jobs[0].next_run_time
        logger.info("Scheduler started — next run at %s", next_run)
    else:
        logger.info("Scheduler started — no jobs queued (unexpected)")

    # Immediate first run
    logger.info("Triggering immediate pipeline run …")
    try:
        run_full_pipeline()
    except Exception as exc:
        logger.error("Immediate pipeline run failed: %s", exc)


def stop_scheduler() -> None:
    """Shut down the background scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Scheduler Smoke Test (single run)")
    print("=" * 70)

    print("\nRunning full pipeline once (no background scheduler) …\n")
    result = run_full_pipeline()

    print("\nPipeline result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
