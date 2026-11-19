"""
embedder.py
-----------
Provides the Embedder class for generating text embeddings.

Primary backend: OpenRouter API (text-embedding-3-small, OpenAI-compatible).
Fallback backend: sklearn TfidfVectorizer + TruncatedSVD (384-dim dense vectors).
Works without an API key during local testing.
"""

import os
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 384
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "text-embedding-3-small"


class Embedder:
    """Generate dense vector embeddings for text chunks.

    Tries OpenRouter (OpenAI-compatible) first; falls back to a local
    TF-IDF + SVD pipeline if the API is unavailable or the key is missing.
    """

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
        self._openai_client = None
        self._tfidf = None          # sklearn TfidfVectorizer
        self._svd = None            # sklearn TruncatedSVD
        self._tfidf_fitted = False

        if self._api_key:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(
                    api_key=self._api_key,
                    base_url=OPENROUTER_BASE_URL,
                )
                logger.info("Embedder: OpenRouter client initialised.")
            except Exception as exc:
                logger.warning("Embedder: failed to initialise OpenAI client: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return a list of 384-dim float vectors, one per input text.

        Args:
            texts: List of raw text strings to embed.

        Returns:
            List of embedding vectors (each a list of 384 floats).
        """
        if not texts:
            return []

        # Sanitise: replace empty strings so the API doesn't error
        sanitised = [t if t.strip() else " " for t in texts]

        if self._openai_client:
            try:
                return self._embed_openrouter(sanitised)
            except Exception as exc:
                logger.warning(
                    "Embedder: OpenRouter embedding failed (%s). Falling back to TF-IDF SVD.",
                    exc,
                )

        return self._embed_tfidf(sanitised)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_openrouter(self, texts: List[str]) -> List[List[float]]:
        """Call OpenRouter embedding endpoint (OpenAI-compatible)."""
        # Batch in chunks of 100 to stay within typical API limits
        results: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
            # Sort by index to preserve order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            results.extend([d.embedding for d in sorted_data])

        # If the model returns vectors of a different dimensionality, pad/trim
        results = [self._resize_vector(v, EMBED_DIM) for v in results]
        return results

    def _embed_tfidf(self, texts: List[str]) -> List[List[float]]:
        """Local TF-IDF + TruncatedSVD fallback (always 384-dim output)."""
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore

        if not self._tfidf_fitted:
            self._tfidf = TfidfVectorizer(
                max_features=8192,
                sublinear_tf=True,
                strip_accents="unicode",
            )
            self._svd = TruncatedSVD(n_components=EMBED_DIM, random_state=42)
            # Fit on the current batch (best effort; more docs = better)
            tfidf_matrix = self._tfidf.fit_transform(texts)
            # SVD needs at least EMBED_DIM+1 columns; pad if necessary
            if tfidf_matrix.shape[1] < EMBED_DIM + 1:
                n_comp = max(1, tfidf_matrix.shape[1] - 1)
                self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            embeddings = self._svd.fit_transform(tfidf_matrix)
            self._tfidf_fitted = True
        else:
            tfidf_matrix = self._tfidf.transform(texts)
            embeddings = self._svd.transform(tfidf_matrix)

        # Normalise to unit length and pad/trim to EMBED_DIM
        result = []
        for vec in embeddings:
            vec = self._resize_vector(vec.tolist(), EMBED_DIM)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            result.append(vec)
        return result

    @staticmethod
    def _resize_vector(vec: List[float], target_dim: int) -> List[float]:
        """Pad with zeros or truncate to reach exactly target_dim."""
        if len(vec) >= target_dim:
            return vec[:target_dim]
        return vec + [0.0] * (target_dim - len(vec))
