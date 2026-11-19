"""
PropTech AI Advisor — Data Ingestion Script
============================================
Run this script to scrape property data from DarGlobal and Wasalt,
then ingest it into the vector database.

Usage:
    python ingest.py

Requirements:
    - OPENROUTER_API_KEY set in .env (or environment)
    - Optionally: POSTGRES_* vars for pgvector; falls back to SQLite
"""

import sys
import time

# ── ANSI colour helpers ────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"

def _print(color: str, prefix: str, msg: str) -> None:
    print(f"{color}{BOLD}{prefix}{RESET} {msg}")

def info(msg: str)    -> None: _print(CYAN,   "[INFO]", msg)
def success(msg: str) -> None: _print(GREEN,  "[OK]  ", msg)
def warn(msg: str)    -> None: _print(YELLOW, "[WARN]", msg)
def error(msg: str)   -> None: _print(RED,    "[ERR] ", msg)

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    info(".env loaded")
except ImportError:
    warn("python-dotenv not installed — reading environment as-is")

# ── Step 1: Scrape DarGlobal ──────────────────────────────────────────────────
info("Starting DarGlobal scraper …")
t0 = time.time()
try:
    from app.scrapers.darglobal_scraper import main as darglobal_main
    darglobal_main()
    success(f"DarGlobal scraping done in {time.time() - t0:.1f}s")
except Exception as exc:
    error(f"DarGlobal scraper failed: {exc}")
    warn("Continuing without DarGlobal data …")

# ── Step 2: Scrape Wasalt ─────────────────────────────────────────────────────
info("Starting Wasalt scraper …")
t1 = time.time()
try:
    from app.scrapers.wasalt_scraper import main as wasalt_main
    wasalt_main()
    success(f"Wasalt scraping done in {time.time() - t1:.1f}s")
except Exception as exc:
    error(f"Wasalt scraper failed: {exc}")
    warn("Continuing without Wasalt data …")

# ── Step 3: Ingest into vector store ─────────────────────────────────────────
info("Running ingestion pipeline (chunk → embed → store) …")
t2 = time.time()
try:
    from app.rag.ingestion import IngestPipeline
    pipeline = IngestPipeline()
    pipeline.run()
    success(f"Ingestion complete in {time.time() - t2:.1f}s")
except Exception as exc:
    error(f"Ingestion failed: {exc}")
    sys.exit(1)

# ── Summary ───────────────────────────────────────────────────────────────────
total = time.time() - t0
print()
success(f"All done in {total:.1f}s — PropTech AI Advisor is ready!")
info("Start the server with:  uvicorn app.main:app --reload")
