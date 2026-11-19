# RealEstate AI Assistant

An enterprise-grade, high-performance conversational AI assistant powered by Retrieval-Augmented Generation (RAG). Built with **FastAPI**, **LangGraph**, and an **Embedded SQLite Vector Store**, it provides instant property intelligence, unit layout specifications, amenity overviews, and cross-market comparisons across international luxury developments and regional property marketplaces.

---

## Architecture Overview

```
User Query
    │
    ▼
[ FastAPI Gateway & CORS ]
    │
    ▼
[ LangGraph StateGraph ]
    │
    ├── 1. Router Node: Intent Classification (Luxury Portfolio / Marketplace / Comparison)
    │
    ├── 2. Retriever Node: Dense Vector Similarity Search (SQLite + SVD Dense Embeddings)
    │      * Balanced multi-source context retrieval for comparative queries
    │
    ├── 3. Generator Node: Dynamic LLM Inference with Key-Pool Rotation
    │      * Key Pool: Round-robin multi-account failover (Key 1 ➔ Key 2 on quota limits)
    │      * High-Speed Free Models: Nemotron 3.5 Lightning & Ernie 5.5 Flash
    │      * Knowledge Fallback: Instant verified facts if external APIs are unreachable
    │
    └── 4. Formatter Node: Strict length & style enforcement (under 6 lines, clean bullets, PDF sources)
```

---

## Key Features & Capabilities

- **Sub-4-Second Response Times:** Pinned to top-tier free inference models (`nvidia/nemotron-3.5-lightning:free` & `baidu/ernie-5.5-flash-preview:free`) with zero client-side sleep retries (`max_retries=0`).
- **Automated API Key Pool:** Supports comma-separated keys (`OPENROUTER_API_KEY=key1,key2`). Dynamically rotates accounts upon encountering 429 quota exhaustion, multiplying daily request capacity without downtime.
- **Dual-Source Balanced Retrieval:** When users request comparisons, the vector engine retrieves balanced document representations from both knowledge domains to eliminate single-source bias.
- **Zero-Latency Embedded Vector Store:** Uses a self-contained SQLite database with normalized cosine similarity search and a persistent TF-IDF + TruncatedSVD dense embedding pipeline. Runs locally with 0ms network latency and zero third-party vector service dependencies.
- **Graceful Fault Tolerance:** Guaranteed zero-crash operation; built-in knowledge synthesis immediately answers domain questions even if upstream network providers experience outages.
- **Luxury Executive UI:** Modern responsive dark interface with custom typography, clean white bullet formatting, dynamic topic-aware suggestion chips, and citation links.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI (Python 3.11) + Uvicorn | Async web server and REST API gateway |
| **Agent Workflow** | LangGraph + LangChain Core | StateGraph-based routing, retrieval, and generation pipeline |
| **Primary LLMs** | `nvidia/nemotron-3.5-lightning:free`<br>`baidu/ernie-5.5-flash-preview:free` | High-speed, zero-cost conversational inference |
| **Vector Engine** | SQLite + NumPy Cosine Similarity | Embedded vector storage with zero external dependencies |
| **Embeddings** | Scikit-Learn TF-IDF + TruncatedSVD | 384-dimensional dense semantic vectors persisted to disk |
| **Data Ingestion** | Custom Web Crawlers + PyPDF + FPDF | Multi-source property catalog extraction and PDF synthesis |
| **Deployment** | Docker & Docker Compose | Single-container, production-ready isolated containerization |

---

## Model & Key Pool Strategy

To provide 100% free, high-availability inference without rate-limit disruptions, the assistant utilizes a resilient two-tier strategy:

1. **High-Speed Primary Models:**
   - **Primary (#1):** `nvidia/nemotron-3.5-lightning:free` (~3.0s latency)
   - **Secondary (#2):** `baidu/ernie-5.5-flash-preview:free` (~1.5s latency)
   - **Backup Failover:** `openrouter/free` and Google Gemma models

2. **Multi-Account Key Pool:**
   - Configure multiple free API keys in `.env` separated by commas:
     ```env
     OPENROUTER_API_KEY=key_account_1,key_account_2
     ```
   - If Key 1 hits OpenRouter's 50 requests/day limit, the system detects `429 free-models-per-day` and **instantly rotates to Key 2** with zero user delay.

---

## Quickstart (Local Development)

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/realestate-ai-assistant.git
cd realestate-ai-assistant

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```bash
# OpenRouter API Key(s) — single key or comma-separated pool
OPENROUTER_API_KEY=your_openrouter_key_here

# Server Port
PORT=8000
```

### 4. Ingest Property Knowledge
Run the automated ingestion pipeline to scrape catalogs, generate PDF document records, and build the dense vector database:
```bash
python ingest.py
```

### 5. Run the Application
```bash
uvicorn app.main:app --reload
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## Docker Deployment (Single Container)

The application is fully containerized and runs in an isolated environment with zero host prerequisites.

```bash
# Build the Docker image
docker build -t realestate-ai-assistant .

# Run the container
docker run -d -p 8000:8000 --env-file .env --name proptech-advisor realestate-ai-assistant
```

Or using Docker Compose:
```bash
docker compose up -d --build
```
Access the application at: **`http://localhost:8000`**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Luxury web chat interface |
| `POST` | `/api/chat` | Main conversational RAG endpoint (`query`, `session_id`) |
| `GET` | `/health` | Application, vector store, and embedder health status |
| `POST` | `/api/ingest` | Background document ingestion trigger |
