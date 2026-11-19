# RealEstate AI Assistant

> 🏠 **AI-powered real estate chatbot using RAG (Retrieval-Augmented Generation)**  
> Answers property questions using live data from DarGlobal and Wasalt, backed by a semantic vector search pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client / Browser                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP / WebSocket
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Application                        │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  /api/chat │  │ /api/health  │  │   /api/ingest       │  │
│  └─────┬──────┘  └──────────────┘  └──────────┬──────────┘  │
│        │                                        │            │
│        ▼                                        ▼            │
│  ┌─────────────────┐              ┌─────────────────────┐   │
│  │  LangGraph RAG  │              │  Ingestion Pipeline │   │
│  │  Agent / Chain  │              │  (Scrape → Chunk →  │   │
│  └────────┬────────┘              │   Embed → Store)    │   │
│           │                       └──────────┬──────────┘   │
│           │                                  │              │
└───────────┼──────────────────────────────────┼──────────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────┐              ┌───────────────────────┐
│  PostgreSQL +     │◄─────────────│  Scrapers             │
│  pgvector         │              │  • DarGlobal (14 URLs)│
│  (vector store)   │              │  • Wasalt (4 URLs)    │
└───────────────────┘              └───────────────────────┘
            │
            ▼
┌───────────────────┐
│  Redis            │
│  (session cache)  │
└───────────────────┘
            │
            ▼
┌───────────────────┐
│  OpenRouter API   │
│  (LLM + Embeds)   │
└───────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI 0.115 + Uvicorn |
| **AI / LLM** | LangChain 0.3 + LangGraph 0.2 via OpenRouter |
| **Embeddings** | OpenAI `text-embedding-3-small` (via OpenRouter) |
| **Vector DB** | PostgreSQL 16 + pgvector (SQLite fallback) |
| **Session Cache** | Redis 7 (in-memory dict fallback) |
| **Scraping** | curl-cffi + BeautifulSoup4 |
| **Data Validation** | Pydantic v2 |
| **Containerization** | Docker + Docker Compose |
| **Deployment** | Render.com (one-click) |

---

## Quickstart — Local (No Docker)

### Prerequisites
- Python 3.11+
- (Optional) PostgreSQL 16 with pgvector extension
- (Optional) Redis 7

```bash
# 1. Clone
git clone https://github.com/your-org/proptech-ai-advisor.git
cd proptech-ai-advisor

# 2. Create virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY

# 5. Scrape + ingest data
python ingest.py

# 6. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Quickstart — Docker Compose

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY and POSTGRES_PASSWORD

# 2. Build and start all services
docker compose up --build

# 3. Run ingestion inside the container (first time only)
docker compose exec app python ingest.py

# App available at http://localhost:8000
# API docs at  http://localhost:8000/docs
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | ✅ Yes | — | OpenRouter API key — get free at [openrouter.ai](https://openrouter.ai) |
| `POSTGRES_HOST` | No | — | PostgreSQL host (omit to use SQLite fallback) |
| `POSTGRES_PORT` | No | `5432` | PostgreSQL port |
| `POSTGRES_DB` | No | `proptech_db` | Database name |
| `POSTGRES_USER` | No | `postgres` | Database user |
| `POSTGRES_PASSWORD` | No | — | Database password |
| `REDIS_URL` | No | — | Redis connection URL (omit to use in-memory fallback) |
| `PORT` | No | `8000` | HTTP port the server listens on |
| `API_SECRET` | No | — | Optional secret for protecting admin endpoints |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/api/chat` | Send a message; returns AI answer with sources |
| `POST` | `/api/ingest` | Trigger scraping + ingestion pipeline |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc UI |

### `POST /api/chat` — Request Body
```json
{
  "message": "Tell me about W Residences Dubai",
  "session_id": "user-abc-123"
}
```

### `POST /api/chat` — Response
```json
{
  "answer": "W Residences Dubai is a luxury ...",
  "sources": [
    {
      "url": "https://darglobal.co.uk/w-residences-dubai",
      "title": "W Residences Dubai",
      "score": 0.92
    }
  ],
  "session_id": "user-abc-123"
}
```

---

## RAG Pipeline

```
Scraping
   │
   ├── DarGlobal (14 property pages via curl-cffi + BeautifulSoup)
   └── Wasalt    (4 pages)
         │
         ▼
   Raw JSON + TXT files saved to data/raw/
         │
         ▼
Chunking (sliding window, ~512 tokens, 50 token overlap)
         │
         ▼
Embedding (text-embedding-3-small via OpenRouter → 384-dim vectors)
         │
         ▼
Storage (pgvector INSERT with UPSERT by URL hash)
         │
         ▼
Retrieval (cosine similarity search, top-k=5)
         │
         ▼
Generation (LangGraph agent → OpenRouter LLM → answer + citations)
```

---

## One-Click Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

The `deploy/render.yaml` is pre-configured. You only need to supply `OPENROUTER_API_KEY` in the Render dashboard.

---

## Project Structure

```
proptech-ai-advisor/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── api/
│   │   └── routes.py        # API route definitions
│   ├── rag/
│   │   ├── embedder.py      # Embedding logic (OpenRouter + TF-IDF fallback)
│   │   ├── vector_store.py  # pgvector / SQLite-vec store
│   │   ├── retriever.py     # Similarity search
│   │   ├── ingestion.py     # Chunk + embed + store pipeline
│   │   └── agent.py         # LangGraph RAG agent
│   └── scrapers/
│       ├── darglobal_scraper.py
│       └── wasalt_scraper.py
├── data/
│   ├── raw/
│   │   ├── darglobal/       # Scraped DarGlobal text files + JSON
│   │   └── wasalt/          # Scraped Wasalt text files + JSON
│   └── processed/
│       └── vector_index/    # Local SQLite vector index (fallback)
├── deploy/
│   ├── init.sql             # PostgreSQL schema (pgvector)
│   └── render.yaml          # Render.com deployment config
├── ingest.py                # Standalone ingestion runner
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## License

MIT © 2024 PropTech AI Advisor
