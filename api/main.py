"""
FastAPI application — Dalal Radar REST API.

Endpoints
---------
POST /chat           — RAG chatbot
POST /feedback       — Submit rating + comment
GET  /stats          — Feedback statistics
GET  /deals          — NSE bulk/block deals
POST /pipeline/run   — Manually trigger data pipeline
GET  /health         — Health check
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap path so sibling packages are importable
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure data directory exists
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create dirs + start scheduler. Shutdown: stop scheduler."""
    # --- startup ---
    os.makedirs(_DATA_DIR, exist_ok=True)
    logger.info("Data directory ensured: %s", _DATA_DIR)

    from pipeline.scheduler import start_scheduler
    logger.info("Starting background scheduler …")
    start_scheduler()

    yield

    # --- shutdown ---
    from pipeline.scheduler import stop_scheduler
    stop_scheduler()
    logger.info("Application shutdown complete")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dalal Radar API",
    description=(
        "Indian stock-market sentiment engine powered by YouTube creator "
        "analysis, NSE deal tracking, and RAG-based Q&A."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    conversation_history: Optional[list[dict]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = ""


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat_endpoint(body: ChatRequest):
    """
    Answer a natural-language question about Indian markets using
    the RAG pipeline.
    """
    from agents.chat_agent import answer_query
    from agents.feedback_agent import log_interaction

    logger.info("POST /chat — query: %s", body.query[:80])

    response = answer_query(
        query=body.query,
        conversation_history=body.conversation_history or [],
    )

    # Log the interaction for feedback tracking
    session_id = log_interaction(
        session_id=body.session_id or "",
        query=body.query,
        response=response,
        retrieved_chunks=[],  # lightweight — full chunks already in ChromaDB
    )

    return {
        "answer": response.get("answer", ""),
        "citations": response.get("citations", []),
        "recommendation": response.get("recommendation"),
        "sources_used": response.get("sources_used", 0),
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

@app.post("/feedback")
def feedback_endpoint(body: FeedbackRequest):
    """
    Submit a user rating (1-5) and optional comment for a chat session.
    Automatically triggers response improvement for low ratings.
    """
    from agents.feedback_agent import submit_feedback

    logger.info(
        "POST /feedback — session=%s rating=%d",
        body.session_id, body.rating,
    )

    result = submit_feedback(
        session_id=body.session_id,
        rating=body.rating,
        comment=body.comment or "",
    )
    return result


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

@app.get("/stats")
def stats_endpoint():
    """Return aggregate feedback statistics."""
    from agents.feedback_agent import get_feedback_stats

    logger.info("GET /stats")
    return get_feedback_stats()


# ---------------------------------------------------------------------------
# GET /deals
# ---------------------------------------------------------------------------

@app.get("/deals")
def deals_endpoint(symbol: Optional[str] = None):
    """
    Return NSE bulk/block deals from the last 24 hours.
    Optionally filter by ticker symbol.
    """
    from agents.market_agent import get_all_deals, get_deals_by_symbol

    logger.info("GET /deals — symbol=%s", symbol)

    if symbol:
        deals = get_deals_by_symbol(symbol)
    else:
        deals = get_all_deals()

    return {"deals": deals, "count": len(deals)}


# ---------------------------------------------------------------------------
# POST /pipeline/run
# ---------------------------------------------------------------------------

@app.post("/pipeline/run")
def pipeline_run_endpoint(background_tasks: BackgroundTasks):
    """Manually trigger a full pipeline run in the background."""
    from pipeline.scheduler import run_full_pipeline

    logger.info("POST /pipeline/run — triggering manual run")

    # Run in background so the request returns immediately
    background_tasks.add_task(run_full_pipeline)

    return {
        "status": "started",
        "message": "Pipeline run triggered in background. Check logs for progress.",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_endpoint():
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "name": "dalal-radar",
    }
