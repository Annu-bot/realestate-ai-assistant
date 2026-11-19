"""
PropTech AI Advisor - Application Entry Point
FastAPI app with lifespan management, middleware, and routing.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env file immediately on startup
load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, set_agent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, tear down on shutdown."""
    logger.info("PropTech AI Advisor starting up …")

    # --- Vector Store ---
    try:
        from app.rag.vector_store import VectorStore  # type: ignore
        vector_store = VectorStore()
        await vector_store.initialize()
        app.state.vector_store = vector_store
        app.state.vector_store_name = vector_store.backend_name  # "pgvector" or "sqlite"
        logger.info("VectorStore initialized: %s", app.state.vector_store_name)
    except Exception as exc:
        logger.warning("VectorStore init failed (%s) — using None placeholder.", exc)
        app.state.vector_store = None
        app.state.vector_store_name = "unavailable"

    # --- Embedder ---
    try:
        from app.rag.embedder import Embedder  # type: ignore
        embedder = Embedder()
        app.state.embedder = embedder
        logger.info("Embedder initialized.")
    except Exception as exc:
        logger.warning("Embedder init failed (%s) — using None placeholder.", exc)
        app.state.embedder = None

    # --- Session History ---
    try:
        from app.rag.session_history import SessionHistory  # type: ignore
        session_history = SessionHistory()
        await session_history.initialize()
        app.state.session_history = session_history
        app.state.history_store_name = session_history.backend_name  # "redis" or "memory"
        logger.info("SessionHistory initialized: %s", app.state.history_store_name)
    except Exception as exc:
        logger.warning("SessionHistory init failed (%s) — using None placeholder.", exc)
        app.state.session_history = None
        app.state.history_store_name = "unavailable"

    # --- PropTech Agent ---
    try:
        from app.rag.agent import PropTechAgent  # type: ignore
        agent = PropTechAgent(
            vector_store=app.state.vector_store,
            embedder=app.state.embedder,
            session_history=app.state.session_history,
        )
        set_agent(agent)
        logger.info("PropTechAgent initialized.")
    except Exception as exc:
        logger.warning("PropTechAgent init failed (%s).", exc)

    logger.info("Startup complete. Serving on http://0.0.0.0:8000")
    yield

    # Shutdown
    logger.info("PropTech AI Advisor shutting down …")
    for attr in ("vector_store", "session_history"):
        obj = getattr(app.state, attr, None)
        if obj and hasattr(obj, "close"):
            try:
                await obj.close()
            except Exception:
                pass
    logger.info("Shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title="RealEstate AI Assistant",
        description=(
            "AI-powered real estate chatbot for DarGlobal luxury properties "
            "and Wasalt Saudi listings."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins for deployment flexibility
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (CSS, JS, images)
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    # API + UI routes
    application.include_router(router)

    return application


app = create_app()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
