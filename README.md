# RealEstate AI Assistant

An intelligent real estate conversational assistant powered by Retrieval-Augmented Generation (RAG). It provides property insights, location comparisons, and residential investment details across luxury developments and regional marketplaces.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend Framework** | FastAPI + Uvicorn |
| **Agent & RAG Orchestration** | LangGraph + LangChain |
| **LLM Provider** | OpenRouter Free-Tier Endpoints |
| **Vector Database** | Embedded SQLite Vector Store (Normalized Cosine Similarity) |
| **Document Processing** | Automated Chunking & Dense Embeddings |
| **Containerization** | Docker + Docker Compose |

---

## Quickstart (Local)

### 1. Prerequisites
- Python 3.11+

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/realestate-ai-assistant.git
cd realestate-ai-assistant

# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file from `.env.example`:
```bash
# Set your OpenRouter API key (get free key at https://openrouter.ai/keys)
OPENROUTER_API_KEY=your_key_here
```

### 4. Ingest Property Knowledge
```bash
python ingest.py
```

### 5. Start the Application
```bash
python -m uvicorn app.main:app --reload
```
Open your browser and navigate to: `http://localhost:8000`

---

## Quickstart (Docker)

Run the entire application in a single isolated container:

```bash
# Build the Docker image
docker build -t realestate-ai-assistant .

# Run container
docker run -p 8000:8000 --env-file .env realestate-ai-assistant
```

Or using Docker Compose:
```bash
docker compose up --build
```
Access the application at: `http://localhost:8000`
