"""
ingestion.py
------------
Document loading, chunking and ingestion pipeline for proptech-ai-advisor.

Usage (standalone):
    python app/rag/ingestion.py

Or import and call:
    from app.rag.ingestion import ingest
    ingest(data_dir, vector_store, embedder)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Chunk sizes (character-based; tiktoken optional)
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

# Try to load tiktoken for token-aware splitting
try:
    import tiktoken  # type: ignore

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    CHUNK_SIZE_TOKENS = 500
    CHUNK_OVERLAP_TOKENS = 50
    _USE_TIKTOKEN = True
    logger.info("ingestion: using tiktoken for chunking.")
except Exception:
    _USE_TIKTOKEN = False
    logger.info("ingestion: tiktoken not available; using char-based chunking.")


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_by_chars(text: str, size: int, overlap: int) -> List[str]:
    """Split *text* into overlapping character-window chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c.strip()]


def _chunk_by_tokens(text: str, size: int, overlap: int) -> List[str]:
    """Split *text* into overlapping token-window chunks using tiktoken."""
    tokens = _TOKENIZER.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunk_tokens = tokens[start:end]
        chunks.append(_TOKENIZER.decode(chunk_tokens))
        start += size - overlap
        if start >= len(tokens):
            break
    return [c for c in chunks if c.strip()]


def _chunk_text(text: str) -> List[str]:
    """Dispatch to the best available chunking strategy."""
    if _USE_TIKTOKEN:
        return _chunk_by_tokens(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
    return _chunk_by_chars(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_txt_file(path: Path, source: str) -> List[Dict[str, Any]]:
    """Load a plain-text file and return one record per file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("ingestion: could not read %s: %s", path, exc)
        return []

    return [
        {
            "text": text,
            "source": source,
            "title": path.stem,
            "url": "",
            "metadata": {"file": str(path), "type": "txt"},
        }
    ]


def _load_pdf_file(path: Path, source: str) -> List[Dict[str, Any]]:
    """Extract text from a PDF file using pypdf/pypdf2 or fallback."""
    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            try:
                raw_bytes = path.read_bytes()
                import re
                strings = re.findall(rb"[a-zA-Z0-9\s.,;:'\"-]{4,}", raw_bytes)
                text = " ".join(s.decode('latin1', errors='ignore') for s in strings)
            except Exception as exc:
                logger.warning("ingestion: could not read PDF %s: %s", path, exc)
                return []
    if not text.strip():
        return []
    return [
        {
            "text": text.strip(),
            "source": source,
            "title": path.stem,
            "url": "",
            "metadata": {"file": str(path), "type": "pdf"},
        }
    ]


def _load_json_file(path: Path, source: str) -> List[Dict[str, Any]]:
    """Load a JSON file.

    Accepts either a single dict or a list of dicts with optional keys:
    title, url, text / description / headings.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("ingestion: could not parse %s: %s", path, exc)
        return []

    if isinstance(raw, dict):
        raw = [raw]

    records: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        # Compose text from known fields
        parts: List[str] = []
        if item.get("title"):
            parts.append(str(item["title"]))
        if item.get("description"):
            parts.append(str(item["description"]))
        if item.get("headings"):
            headings = item["headings"]
            if isinstance(headings, list):
                parts.extend(
                    h.get("text", str(h)) if isinstance(h, dict) else str(h)
                    for h in headings
                )
            else:
                parts.append(str(headings))
        if item.get("text"):
            text_val = item["text"]
            if isinstance(text_val, list):
                parts.extend(str(t) for t in text_val)
            else:
                parts.append(str(text_val))

        combined = "\n".join(parts).strip()
        if not combined:
            continue

        records.append(
            {
                "text": combined,
                "source": source,
                "title": item.get("title", path.stem),
                "url": item.get("url", ""),
                "metadata": {
                    "file": str(path),
                    "type": "json",
                    "scraped_at": item.get("scraped_at", ""),
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_chunk_documents(data_dir: str) -> List[Dict[str, Any]]:
    """Load raw data files and split them into overlapping text chunks.

    Scans ``<data_dir>/data/raw/darglobal/`` and
    ``<data_dir>/data/raw/wasalt/`` for ``.txt`` and ``.json`` files.

    Args:
        data_dir: Root directory of the project (contains ``data/raw/``).

    Returns:
        List of chunk dicts, each with keys:
        ``text``, ``source``, ``title``, ``url``, ``chunk_index``,
        ``metadata``.
    """
    base = Path(data_dir)
    sources = {
        "darglobal": base / "data" / "raw" / "darglobal",
        "wasalt": base / "data" / "raw" / "wasalt",
    }

    all_chunks: List[Dict[str, Any]] = []

    for source_name, raw_dir in sources.items():
        if not raw_dir.exists():
            logger.warning("ingestion: directory not found, skipping: %s", raw_dir)
            continue

        records: List[Dict[str, Any]] = []
        for path in sorted(raw_dir.iterdir()):
            if path.suffix.lower() == ".txt":
                records.extend(_load_txt_file(path, source_name))
            elif path.suffix.lower() == ".json":
                records.extend(_load_json_file(path, source_name))
            elif path.suffix.lower() == ".pdf":
                records.extend(_load_pdf_file(path, source_name))

        logger.info(
            "ingestion: loaded %d record(s) from %s", len(records), source_name
        )

        for record in records:
            text = record["text"]
            chunks = _chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "source": record["source"],
                        "title": record["title"],
                        "url": record["url"],
                        "chunk_index": idx,
                        "metadata": {
                            **record["metadata"],
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                        },
                    }
                )

    logger.info("ingestion: produced %d total chunks.", len(all_chunks))
    return all_chunks


def ingest(
    data_dir: str,
    vector_store: Any,
    embedder: Any,
    batch_size: int = 32,
) -> None:
    """Full ingestion pipeline: load → chunk → embed → store.

    Args:
        data_dir: Root project directory.
        vector_store: A ``VectorStore`` instance.
        embedder: An ``Embedder`` instance.
        batch_size: Number of chunks to embed and store at a time.
    """
    chunks = load_and_chunk_documents(data_dir)
    total = len(chunks)
    if total == 0:
        print("ingestion: no chunks found – nothing to ingest.")
        return

    # Clear previous chunks for clean idempotent ingestion
    if hasattr(vector_store, "clear"):
        vector_store.clear()

    # Pre-fit embedder across full corpus vocabulary
    if hasattr(embedder, "fit_corpus"):
        all_texts = [c["text"] for c in chunks]
        embedder.fit_corpus(all_texts)

    print(f"ingestion: {total} chunks to embed and store (batch_size={batch_size}).")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]

        embeddings = embedder.embed_texts(texts)

        docs_to_store = []
        for chunk, embedding in zip(batch, embeddings):
            docs_to_store.append({**chunk, "embedding": embedding})

        vector_store.add_documents(docs_to_store)

        end_idx = min(batch_start + batch_size, total)
        print(
            f"  [{end_idx}/{total}] stored chunks "
            f"{batch_start + 1}–{end_idx}  "
            f"(backend: {vector_store.health_check()})"
        )

    print("ingestion: done.")


class IngestPipeline:
    """Convenience pipeline runner used by background tasks and API routes."""

    def __init__(
        self,
        vector_store: Any = None,
        embedder: Any = None,
        data_dir: Optional[str] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        if data_dir is None:
            self.data_dir = str(Path(__file__).resolve().parents[2])
        else:
            self.data_dir = data_dir

    def run(self) -> None:
        """Run the ingestion pipeline."""
        if self.vector_store is None:
            from app.rag.vector_store import VectorStore
            self.vector_store = VectorStore()
        if self.embedder is None:
            from app.rag.embedder import Embedder
            self.embedder = Embedder()
        ingest(self.data_dir, self.vector_store, self.embedder)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Resolve project root (two levels up from this file)
    _HERE = Path(__file__).resolve()
    _PROJECT_ROOT = str(_HERE.parents[2])

    print(f"Project root: {_PROJECT_ROOT}")

    from app.rag.embedder import Embedder  # noqa: E402
    from app.rag.vector_store import VectorStore  # noqa: E402

    _embedder = Embedder()
    _store = VectorStore()

    print(f"Vector store backend: {_store.health_check()}")

    try:
        ingest(_PROJECT_ROOT, _store, _embedder)
    finally:
        _store.close()
