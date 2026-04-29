# 🎯 Dalal Radar

**AI-powered NSE/BSE sentiment analysis from 100 Indian YouTube trading creators**

Dalal Radar is an autonomous intelligence engine that scrapes Indian stock market YouTube creators, extracts ticker-level sentiment using LLMs, combines it with live NSE bulk/block deal data, and serves it all through a RAG-powered chatbot with citations and trading recommendations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DALAL RADAR ENGINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────────────┐  │
│  │  YouTube     │     │  Market      │    │  Feedback            │  │
│  │  Agent       │     │  Agent       │    │  Agent               │  │
│  │              │     │              │    │                      │  │
│  │  • Apify     │     │  • nselib    │    │  • Log interactions  │  │
│  │  • 100 videos│     │  • jugaad    │    │  • Accept ratings    │  │
│  │ • Transcripts│     │  • httpx     │    │  • LLM rewrite on    │  │
│  │  • LLM       │     │  • Bulk/Block│    │    poor feedback     │  │
│  │    sentiment │     │    deals     │    │  • Stats dashboard   │  │
│  └──────┬───────┘     └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       │               │
│  ┌────────────────────────────────┐                 │               │
│  │         PIPELINE               │                 │               │
│  │                                │                 │               │
│  │  Chunker ──► Embedder ──► ChromaDB               │               │
│  │  (sentence   (MiniLM-L6)  (persistent)           │               │
│  │   boundaries, 384-dim      cosine                │               │
│  │   400 tokens)  vectors     similarity)           │               │
│  └────────────────┬───────────────┘                 │               │
│                   │                                 │               │
│                   ▼                                 │               │
│  ┌────────────────────────────────────────────────┐ │               │
│  │              CHAT AGENT (RAG)                  │◄┘               │
│  │                                                │                 │
│  │  Query ──► Ticker Extract ──► Vector Search    │                 │
│  │         ──► Context Build  ──► LLM Answer      │                 │
│  │         ──► Citation Parse ──► Recommendation  │                 │
│  └────────────────┬───────────────────────────────┘                 │
│                   │                                                 │
│                   ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    FastAPI REST API                            │ │
│  │  POST /chat  ·  POST /feedback  ·  GET /stats  ·  GET /deals │ │
│  │  POST /pipeline/run  ·  GET /health                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Scheduler (APScheduler · 24h cycle · auto-refresh)           │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- [Apify](https://apify.com) account (free tier works)
- [OpenRouter](https://openrouter.ai) API key (free models available)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/dalal-radar.git
cd dalal-radar
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
APIFY_TOKEN=your_apify_token_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 4. Run the data pipeline (first-time ingestion)

```bash
python -m pipeline.scheduler
```

This fetches live NSE deals and YouTube sentiment data, chunks it, embeds it, and stores everything in ChromaDB. Takes 5–15 minutes depending on API speeds.

### 5. Start the API server

```bash
uvicorn api.main:app --reload
```

The server starts at `http://localhost:8000` with auto-docs at `/docs`.

> **Note:** The 24-hour background scheduler starts automatically with the API server, so the data stays fresh without manual re-runs.

---

## 📡 API Reference

### `POST /chat` — Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the sentiment on RELIANCE?"}'
```

### `POST /feedback` — Rate a response

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "rating": 5, "comment": "Very accurate!"}'
```

### `GET /stats` — Feedback analytics

```bash
curl http://localhost:8000/stats
```

### `GET /deals` — NSE bulk/block deals

```bash
# All deals from the last 24 hours
curl http://localhost:8000/deals

# Filter by ticker
curl http://localhost:8000/deals?symbol=RELIANCE
```

### `POST /pipeline/run` — Manual pipeline trigger

```bash
curl -X POST http://localhost:8000/pipeline/run
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0", "name": "dalal-radar"}
```

---

## 💬 Sample Chat Response

```json
{
  "answer": "Based on recent YouTube analysis, RELIANCE is showing bullish sentiment. MarketGuru noted strong quarterly earnings and upward momentum on the daily chart [MarketGuru, 2026-04-25]. TradingView Hindi confirmed RSI above 60 and positive MACD crossover [TradingView Hindi, https://youtube.com/watch?v=...]. However, StockTalks maintained a neutral stance citing FII selling pressure [StockTalks, 2026-04-25]. On the deals front, a bulk deal of 2,50,000 shares was recorded at ₹2,850.00 by HDFC Mutual Fund.",
  "citations": [
    {"channel": "MarketGuru", "symbol": "RELIANCE", "sentiment": "bullish", "video_url": "https://youtube.com/watch?v=..."},
    {"channel": "TradingView Hindi", "symbol": "RELIANCE", "sentiment": "bullish", "video_url": "https://youtube.com/watch?v=..."},
    {"channel": "StockTalks", "symbol": "RELIANCE", "sentiment": "neutral", "video_url": "https://youtube.com/watch?v=..."}
  ],
  "recommendation": {
    "symbol": "RELIANCE",
    "direction": "buy",
    "confidence": "medium",
    "reasoning": "3 out of 4 sources are bullish on RELIANCE with strong technical indicators...",
    "disclaimer": "Not financial advice. Always do your own research."
  },
  "sources_used": 7,
  "session_id": "a1b2c3d4e5f6"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | OpenRouter (Mistral-7B-Instruct) | Sentiment extraction, ticker parsing, answer generation |
| **Scraping** | Apify (youtube-scraper, transcript-scraper) | YouTube video discovery + transcript fetching |
| **Market Data** | nselib · jugaad-data · httpx | NSE bulk/block deals with triple-fallback |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim vector embeddings for semantic search |
| **Vector DB** | ChromaDB (persistent) | Cosine similarity retrieval with metadata filtering |
| **API** | FastAPI + Uvicorn | REST endpoints with Pydantic validation |
| **Scheduler** | APScheduler | 24-hour automated pipeline refresh |
| **Data Processing** | Pandas · NumPy | Deal normalization and data wrangling |

---

## 📁 Project Structure

```
dalal-radar/
├── agents/
│   ├── __init__.py
│   ├── chat_agent.py          # RAG chatbot with citations + recommendations
│   ├── feedback_agent.py      # Closed learning loop with LLM rewrite
│   ├── market_agent.py        # NSE bulk/block deal fetcher (3 fallbacks)
│   └── youtube_agent.py       # YouTube scraper + LLM sentiment pipeline
│
├── pipeline/
│   ├── __init__.py
│   ├── chunker.py             # Sentence-boundary text chunking
│   ├── embedder.py            # MiniLM-L6-v2 embedding (cached singleton)
│   └── scheduler.py           # 24h APScheduler + 7-step orchestrator
│
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py           # (reserved for future retrieval strategies)
│   └── vector_store.py        # ChromaDB wrapper with search + filtering
│
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI app with 6 REST endpoints
│
├── data/
│   └── feedback_store.json    # Persistent feedback/session log
│
├── chroma_db/                 # ChromaDB persistent storage (auto-created)
├── config.py                  # Central config: env vars + logging
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md
```

---

## 🔄 How It Works

1. **Every 24 hours**, the scheduler triggers the full pipeline:
   - Scrapes ~100 Indian trading YouTube videos published in the last 24h
   - Fetches transcripts via Apify
   - Runs Mistral-7B to extract ticker-level sentiment (bullish/bearish/neutral)
   - Fetches NSE bulk & block deals with a 3-strategy fallback (nselib → jugaad-data → raw httpx)
   - Chunks everything on sentence boundaries (~400 tokens, 50-token overlap)
   - Generates 384-dim embeddings and stores in ChromaDB

2. **When a user asks a question**, the chat agent:
   - Extracts mentioned tickers from the query using LLM
   - Retrieves relevant chunks via cosine similarity + metadata filtering
   - Generates a grounded answer citing specific channels and URLs
   - Produces a trading recommendation only when 3+ sources agree on sentiment

3. **Feedback loop**: Users rate responses 1–5. Poor ratings auto-trigger an LLM rewrite using the original context, creating a self-improving system.

---

## 🧪 Testing Individual Modules

Each module has a `__main__` block for standalone testing:

```bash
# Test market data fetching (hits live NSE)
python -m agents.market_agent

# Test YouTube pipeline (requires APIFY_TOKEN)
python -m agents.youtube_agent

# Test chunker with sample data
python -m pipeline.chunker

# Test embedder + vector store end-to-end
python -m retrieval.vector_store

# Test chat agent with 3 sample queries
python -m agents.chat_agent

# Test feedback persistence
python -m agents.feedback_agent
```

---

## 📜 License

This project is for educational and assessment purposes.

---

<p align="center">
  <em>Built for the <strong>CrowdWisdomTrading</strong> internship assessment</em><br>
  <sub>Dalal Radar — turning crowd wisdom into market intelligence 📊</sub>
</p>
