"""
embedder.py
-----------
Provides the Embedder class for generating text embeddings.

Primary backend: Embedded local TF-IDF + TruncatedSVD pipeline (384-dim dense vectors).
Fitted across the entire property corpus and persisted to disk for zero-latency,
100% deterministic, offline search. Completely free with zero external API dependencies.

Optional backend: OpenRouter / OpenAI API embeddings when USE_OPENROUTER_EMBEDDINGS=1.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 384
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "vector_index"
MODEL_PATH = MODEL_DIR / "tfidf_model.pkl"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "text-embedding-3-small"


class Embedder:
    """Generate dense vector embeddings for text chunks.

    Uses an embedded scikit-learn TF-IDF + TruncatedSVD pipeline persisted to disk,
    guaranteeing that runtime queries and indexed property documents exist in the exact
    same mathematical vector space with zero API cost and zero rate limits.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or MODEL_PATH
        self._api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
        self._use_api_embeddings = os.environ.get("USE_OPENROUTER_EMBEDDINGS", "0") == "1"
        self._openai_client = None

        self._tfidf = None
        self._svd = None
        self._tfidf_fitted = False

        # Load persisted model if present on disk
        self._load_model()

        if self._use_api_embeddings and self._api_key:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(
                    api_key=self._api_key,
                    base_url=OPENROUTER_BASE_URL,
                )
                logger.info("Embedder: OpenRouter client initialized for embeddings.")
            except Exception as exc:
                logger.warning("Embedder: failed to initialize OpenAI client: %s", exc)

    def _load_model(self) -> bool:
        """Load persisted TF-IDF and SVD models from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self._tfidf = data.get("tfidf")
                    self._svd = data.get("svd")
                    self._tfidf_fitted = True
                    logger.info("Embedder: Loaded fitted embedding model from %s", self.model_path)
                    return True
            except Exception as exc:
                logger.warning("Embedder: Could not load saved model (%s), will re-fit.", exc)
        return False

    def _save_model(self) -> None:
        """Persist TF-IDF and SVD models to disk."""
        try:
            os.makedirs(self.model_path.parent, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump({"tfidf": self._tfidf, "svd": self._svd}, f)
            logger.info("Embedder: Persisted embedding model to %s", self.model_path)
        except Exception as exc:
            logger.warning("Embedder: Failed to save model: %s", exc)

    def fit_corpus(self, texts: List[str]) -> None:
        """Fit TF-IDF and SVD across the entire corpus for optimal vocabulary coverage."""
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore

        if not texts:
            return

        sanitized = [t if t.strip() else " " for t in texts]

        self._tfidf = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf_matrix = self._tfidf.fit_transform(sanitized)

        n_comp = min(EMBED_DIM, max(2, tfidf_matrix.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(tfidf_matrix)

        self._tfidf_fitted = True
        self._save_model()
        logger.info(
            "Embedder: Fitted corpus (%d documents, %d features, %d components).",
            len(texts),
            tfidf_matrix.shape[1],
            n_comp,
        )

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

        sanitized = [t if t.strip() else " " for t in texts]

        if self._use_api_embeddings and self._openai_client:
            try:
                return self._embed_openrouter(sanitized)
            except Exception as exc:
                logger.warning(
                    "Embedder: OpenRouter embedding failed (%s). Falling back to local TF-IDF SVD.",
                    exc,
                )

        return self._embed_tfidf(sanitized)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_openrouter(self, texts: List[str]) -> List[List[float]]:
        """Call OpenRouter embedding endpoint (OpenAI-compatible)."""
        results: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda d: d.index)
            results.extend([d.embedding for d in sorted_data])

        results = [self._resize_vector(v, EMBED_DIM) for v in results]
        return results

    def _embed_tfidf(self, texts: List[str]) -> List[List[float]]:
        """Local TF-IDF + TruncatedSVD transformation (always 384-dim output)."""
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore

        if not self._tfidf_fitted:
            # Try loading from disk one more time
            if not self._load_model():
                self.fit_corpus(texts)

        try:
            tfidf_matrix = self._tfidf.transform(texts)
            embeddings = self._svd.transform(tfidf_matrix)
        except Exception:
            # Emergency fallback fit
            self.fit_corpus(texts)
            tfidf_matrix = self._tfidf.transform(texts)
            embeddings = self._svd.transform(tfidf_matrix)

        result = []
        for vec in embeddings:
            v_list = self._resize_vector(vec.tolist(), EMBED_DIM)
            norm = float(np.linalg.norm(v_list))
            if norm > 0:
                v_list = [v / norm for v in v_list]
            result.append(v_list)
        return result

    @staticmethod
    def _resize_vector(vec: List[float], target_dim: int) -> List[float]:
        """Pad with zeros or truncate to reach exactly target_dim."""
        if len(vec) >= target_dim:
            return vec[:target_dim]
        return vec + [0.0] * (target_dim - len(vec))
