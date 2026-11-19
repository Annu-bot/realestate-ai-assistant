"""
PropTech AI Advisor - API Routes
FastAPI router handling all endpoints for the chat interface.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Pydantic Models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: list
    intent: str
    session_id: str
    suggested_questions: Optional[list] = None


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    history_store: str


# ── Lazy agent reference (populated from main.py on startup) ─────────────────
# We import PropTechAgent lazily to avoid circular imports and allow
# the lifespan handler in main.py to inject the initialized instance.

_agent = None  # type: ignore

def set_agent(agent) -> None:
    """Called by main.py startup to inject the initialized PropTechAgent."""
    global _agent
    _agent = agent


def get_agent():
    """Return the shared PropTechAgent; raises if not yet initialized."""
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not yet initialized. Please retry in a moment.",
        )
    return _agent


# ── Helper: templates directory ───────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_index():
    """Serve the main chat UI (index.html)."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@router.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the PropTech AI Advisor.

    - **session_id**: Unique session identifier (UUID from client)
    - **message**: User's natural-language question
    """
    agent = get_agent()
    try:
        result = agent.chat(session_id=request.session_id, query=request.message)
        return ChatResponse(
            response=result.get("response", ""),
            sources=result.get("sources", []),
            intent=result.get("intent", "general"),
            session_id=request.session_id,
            suggested_questions=result.get("suggested_questions", []),
        )
    except Exception as exc:
        logger.exception("Error during chat: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/health", response_model=HealthResponse, tags=["Operations"])
async def health(request: Request):
    """
    Health check endpoint.
    Returns current vector-store and session-history backend names.
    """
    # Read backend names injected into app.state by main.py
    state = request.app.state
    vector_store_name = getattr(state, "vector_store_name", "sqlite")
    history_store_name = getattr(state, "history_store_name", "memory")
    return HealthResponse(
        status="ok",
        vector_store=vector_store_name,
        history_store=history_store_name,
    )


@router.post("/api/ingest", tags=["Operations"])
async def ingest(
    background_tasks: BackgroundTasks,
    request: Request,
    x_api_secret: Optional[str] = Header(default=None),
):
    """
    Trigger the ingestion pipeline (admin-only).
    Requires the `X-Api-Secret` header matching the `API_SECRET` env var.
    """
    expected_secret = os.getenv("API_SECRET", "changeme")
    if x_api_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing API secret.")

    state = request.app.state
    vector_store = getattr(state, "vector_store", None)
    embedder = getattr(state, "embedder", None)

    if vector_store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Stores not initialized yet.")

    def run_ingestion():
        """Run ingestion in background."""
        try:
            from app.rag.ingestion import IngestPipeline  # type: ignore
            pipeline = IngestPipeline(vector_store=vector_store, embedder=embedder)
            pipeline.run()
            logger.info("Ingestion pipeline completed successfully.")
        except Exception as exc:
            logger.exception("Ingestion pipeline failed: %s", exc)

    background_tasks.add_task(run_ingestion)
    return JSONResponse({"status": "ingestion_started"})


@router.delete("/api/session/{session_id}", tags=["Chat"])
async def clear_session(session_id: str, request: Request):
    """
    Clear the chat history for a given session.
    """
    state = request.app.state
    session_history = getattr(state, "session_history", None)
    if session_history is None:
        raise HTTPException(status_code=503, detail="Session store not initialized.")
    try:
        session_history.clear(session_id)
        return JSONResponse({"status": "cleared", "session_id": session_id})
    except Exception as exc:
        logger.exception("Error clearing session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
