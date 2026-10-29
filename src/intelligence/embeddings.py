"""Semantic Embedding Manager with Sentence Transformers and TF-IDF Fallback.

Generates normalized semantic embeddings and computes cosine similarity.
Gracefully falls back to TF-IDF vectorization if pretrained neural models
are unavailable, explicitly identifying the active method.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


class SemanticEmbeddingManager:
    """Manages sentence embedding generation and pairwise semantic similarity."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        self.active_method = "uninitialized"
        self._initialize_encoder()

    def _initialize_encoder(self) -> None:
        """Attempt to load SentenceTransformer model; fallback to TF-IDF on failure."""
        try:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model '%s' on CPU...", self.model_name)
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.active_method = f"sentence_transformers ({self.model_name})"
            logger.info("SentenceTransformer successfully loaded.")
            return
        except Exception as e:
            logger.warning(
                "Could not load SentenceTransformer ('%s'): %s. Initializing TF-IDF fallback.",
                self.model_name,
                e,
            )

        # TF-IDF Fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2500,
                ngram_range=(1, 2),
                stop_words="english",
            )
            self.active_method = "tfidf_fallback"
            logger.info("TF-IDF vectorizer fallback initialized.")
        except Exception as e2:
            self.active_method = "lexical_jaccard_fallback"
            logger.warning("scikit-learn not available. Using lexical Jaccard fallback: %s", e2)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into normalized 2D numpy embedding vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Clean empty texts
        safe_texts = [t.strip() if t and t.strip() else "empty text" for t in texts]

        if "sentence_transformers" in self.active_method and self.model is not None:
            embeddings = self.model.encode(
                safe_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)

        elif self.active_method == "tfidf_fallback" and self.tfidf_vectorizer is not None:
            # Fit or transform on provided texts
            try:
                tfidf_mat = self.tfidf_vectorizer.fit_transform(safe_texts).toarray()
                # L2 normalize rows
                norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (tfidf_mat / norms).astype(np.float32)
            except Exception as e:
                logger.warning("TF-IDF encoding error: %s", e)

        # Lexical dummy vectors if all else fails
        return np.ones((len(texts), 10), dtype=np.float32) / np.sqrt(10)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts in [0.0, 1.0]."""
        if not text_a or not text_b:
            return 0.0

        if text_a.strip().lower() == text_b.strip().lower():
            return 1.0

        vecs = self.encode_texts([text_a, text_b])
        if len(vecs) < 2:
            return 0.0

        sim = float(np.dot(vecs[0], vecs[1]))
        return max(0.0, min(1.0, sim))

    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute N x N pairwise cosine similarity matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vecs = self.encode_texts(texts)
        # Cosine similarity for unit-normalized vectors is the dot product matrix
        sim_matrix = np.dot(vecs, vecs.T)
        return np.clip(sim_matrix, 0.0, 1.0)
