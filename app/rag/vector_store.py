"""
vector_store.py
---------------
Embedded, zero-latency vector store for PropTech AI Advisor.

Engineered with a self-contained SQLite database + NumPy normalized cosine
similarity search. Runs identically across local development, Docker containers,
and cloud deployments with 0ms network overhead and zero external dependencies.
"""

import json
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQLITE_DB_PATH = str(PROJECT_ROOT / "data" / "processed" / "vector_index" / "index.db")
EMBED_DIM = 384
TABLE_NAME = "proptech_docs"


class VectorStore:
    """High-performance, embedded vector store with normalized cosine similarity search."""

    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._backend_name = "sqlite-embedded"
        self._ensure_table()
        logger.info("VectorStore initialized @ %s", db_path)

    @property
    def backend_name(self) -> str:
        """Return active vector backend name."""
        return self._backend_name

    async def initialize(self) -> None:
        """Async initialization hook for lifespan compatibility."""
        pass

    def _ensure_table(self) -> None:
        """Create docs table and index if they do not exist."""
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id         TEXT PRIMARY KEY,
                source     TEXT,
                title      TEXT,
                url        TEXT,
                chunk_text TEXT,
                metadata   TEXT,
                embedding  TEXT
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_source ON {TABLE_NAME}(source)"
        )
        self._conn.commit()

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Persist document chunks with their dense vector embeddings.

        Args:
            docs: List of dicts with keys: text, source, title, url, metadata, embedding.
        """
        for doc in docs:
            embedding = doc.get("embedding", [])
            chunk_text = doc.get("text") or doc.get("chunk_text") or ""
            self._conn.execute(
                f"""
                INSERT OR IGNORE INTO {TABLE_NAME}
                    (id, source, title, url, chunk_text, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    doc.get("source", ""),
                    doc.get("title", ""),
                    doc.get("url", ""),
                    chunk_text,
                    json.dumps(doc.get("metadata", {})),
                    json.dumps(embedding),
                ),
            )
        self._conn.commit()

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find the top-k most relevant documents via normalized cosine similarity.

        Args:
            query_embedding: Dense vector of the query (e.g. 384-dim).
            k: Number of results to return.
            source_filter: Optional source filter ('darglobal' or 'wasalt').

        Returns:
            List of dicts with keys: id, source, title, url, chunk_text, metadata, similarity.
        """
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        sql = f"SELECT id, source, title, url, chunk_text, metadata, embedding FROM {TABLE_NAME}"
        params: list = []
        if source_filter and source_filter.lower() in ("darglobal", "wasalt"):
            sql += " WHERE LOWER(source) = ?"
            params.append(source_filter.lower())

        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        scored: List[tuple] = []
        for row in rows:
            try:
                raw_embed = row["embedding"]
                if not raw_embed:
                    continue
                vec = np.array(json.loads(raw_embed), dtype=np.float32)
                v_norm = np.linalg.norm(vec)
                if v_norm > 0:
                    vec = vec / v_norm
                similarity = float(np.dot(q, vec))
            except Exception:
                similarity = 0.0
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for sim, row in scored[:k]:
            d = dict(row)
            try:
                d["metadata"] = json.loads(d.get("metadata", "{}"))
            except Exception:
                d["metadata"] = {}
            d["similarity"] = sim
            d.pop("embedding", None)
            results.append(d)

        return results

    def health_check(self) -> str:
        """Return the health status and engine name."""
        return "sqlite-embedded"

    def close(self) -> None:
        """Close SQLite database connection safely."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
