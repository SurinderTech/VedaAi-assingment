"""
Step 8 — Semantic Intelligence & Embedding-Based Candidate Retrieval Service.

Provides dense vector embeddings using SentenceTransformers with dictionary caching,
batch processing, text normalization, and graceful TF-IDF fallback.

STRICT SAFEGUARDS:
- Does NOT fail or crash if SentenceTransformers model is uninstalled/offline (reports UNAVAILABLE).
- Preserves existing TF-IDF similarity_matrix implementation unchanged as explicit fallback path.
- Does NOT mutate Step 1-7 result states or grading decisions.
"""
from __future__ import annotations
import hashlib
import re
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from app.core.config import settings

# Global Model & Cache State
_MODEL_INSTANCE: Optional[Any] = None
_MODEL_LOAD_ATTEMPTED: bool = False
_MODEL_STATUS: str = "UNINITIALIZED" # "AVAILABLE", "UNAVAILABLE", "UNINITIALIZED"
_MODEL_DIMENSION: int = 384

_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}
_CACHE_STATS: Dict[str, int] = {
    "embeddings_requested": 0,
    "embeddings_computed": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}


def _clean_text_for_embedding(text: str) -> str:
    """
    Normalizes text for embedding without destroying technical terminology,
    mathematical equations (e.g. ReLU(-5), x^2 + 2x + 1), question numbers, or hyphenated words.
    """
    if not text:
        return ""
    # Normalize excessive whitespace while preserving hyphens, math operators, and alphanumerics
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned


def _get_model() -> Tuple[Optional[Any], str]:
    """Lazy-loads SentenceTransformer model. Returns (model, status)."""
    global _MODEL_INSTANCE, _MODEL_LOAD_ATTEMPTED, _MODEL_STATUS, _MODEL_DIMENSION

    if not settings.EMBEDDING_ENGINE_ENABLED:
        _MODEL_STATUS = "UNAVAILABLE"
        return None, "UNAVAILABLE"

    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_INSTANCE, _MODEL_STATUS

    _MODEL_LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL_INSTANCE = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        _MODEL_STATUS = "AVAILABLE"
        # Determine vector dimension
        dummy_vec = _MODEL_INSTANCE.encode("test", show_progress_bar=False)
        _MODEL_DIMENSION = len(dummy_vec) if hasattr(dummy_vec, "__len__") else 384
        print(f"[EmbeddingService] SentenceTransformer model '{settings.EMBEDDING_MODEL_NAME}' loaded successfully ({_MODEL_DIMENSION}-dim).")
    except Exception as e:
        print(f"[EmbeddingService] SentenceTransformers loading notice: {e}. Falling back to TF-IDF.")
        _MODEL_INSTANCE = None
        _MODEL_STATUS = "UNAVAILABLE"

    return _MODEL_INSTANCE, _MODEL_STATUS


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """
    Encodes a list of text strings into normalized dense vector embeddings using SentenceTransformers with cache.
    Returns np.ndarray of shape (N, dim), or None if embedding model is unavailable.
    """
    if not texts:
        return np.zeros((0, _MODEL_DIMENSION))

    model, status = _get_model()
    if status != "AVAILABLE" or model is None:
        return None

    cleaned_texts = [_clean_text_for_embedding(t) for t in texts]
    n = len(cleaned_texts)
    embeddings = np.zeros((n, _MODEL_DIMENSION), dtype=np.float32)

    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    model_name = settings.EMBEDDING_MODEL_NAME

    for idx, txt in enumerate(cleaned_texts):
        _CACHE_STATS["embeddings_requested"] += 1
        cache_key = hashlib.sha256(f"{model_name}:{txt}".encode("utf-8")).hexdigest()

        if settings.EMBEDDING_CACHE_ENABLED and cache_key in _EMBEDDING_CACHE:
            embeddings[idx] = _EMBEDDING_CACHE[cache_key]
            _CACHE_STATS["cache_hits"] += 1
        else:
            _CACHE_STATS["cache_misses"] += 1
            uncached_indices.append(idx)
            uncached_texts.append(txt)

    if uncached_texts:
        try:
            vecs = model.encode(
                uncached_texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                normalize_embeddings=settings.EMBEDDING_NORMALIZE,
                show_progress_bar=False,
            )
            vecs = np.array(vecs, dtype=np.float32)

            for i, vec in enumerate(vecs):
                target_idx = uncached_indices[i]
                txt = uncached_texts[i]
                cache_key = hashlib.sha256(f"{model_name}:{txt}".encode("utf-8")).hexdigest()

                embeddings[target_idx] = vec
                if settings.EMBEDDING_CACHE_ENABLED:
                    _EMBEDDING_CACHE[cache_key] = vec
                _CACHE_STATS["embeddings_computed"] += 1
        except Exception as e:
            print(f"[EmbeddingService] Batch encoding exception: {e}")
            return None

    return embeddings


def embed_text(text: str) -> Optional[np.ndarray]:
    """Encodes a single text string into a 1D vector numpy array."""
    res = embed_texts([text])
    if res is not None and len(res) > 0:
        return res[0]
    return None


def cosine_similarity_matrix(query_vecs: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    """Calculates cosine similarity matrix between query vectors (N x d) and candidate vectors (M x d)."""
    if query_vecs is None or candidate_vecs is None or len(query_vecs) == 0 or len(candidate_vecs) == 0:
        return np.zeros((0, 0))

    sim = sklearn_cosine_similarity(query_vecs, candidate_vecs)
    return np.clip(sim, 0.0, 1.0)


def similarity_matrix_tfidf(questions: List[str], answers: List[str]) -> np.ndarray:
    """Exact lexical TF-IDF + Cosine Similarity fallback path (preserved 100% unchanged)."""
    if not questions or not answers:
        return np.zeros((len(questions), len(answers)))

    cleaned_q = [_clean_text_for_embedding(q) for q in questions]
    cleaned_a = [_clean_text_for_embedding(a) for a in answers]

    corpus = cleaned_q + cleaned_a
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    try:
        tfidf = vectorizer.fit_transform(corpus)
        q_vecs = tfidf[: len(questions)]
        a_vecs = tfidf[len(questions):]
        sim = sklearn_cosine_similarity(q_vecs, a_vecs)
        return np.clip(sim, 0.0, 1.0)
    except Exception:
        return np.zeros((len(questions), len(answers)))


def similarity_matrix(questions: List[str], answers: List[str]) -> np.ndarray:
    """
    Returns an (len(questions) x len(answers)) matrix of similarity scores in [0, 1].
    Uses SentenceTransformers dense embeddings if AVAILABLE; falls back seamlessly to TF-IDF.
    """
    if not questions or not answers:
        return np.zeros((len(questions), len(answers)))

    q_vecs = embed_texts(questions)
    a_vecs = embed_texts(answers)

    if q_vecs is not None and a_vecs is not None:
        return cosine_similarity_matrix(q_vecs, a_vecs)

    # Fallback to unchanged TF-IDF implementation
    return similarity_matrix_tfidf(questions, answers)


def get_model_metadata() -> Dict[str, Any]:
    """Exposes metadata about current embedding model status."""
    _, status = _get_model()
    return {
        "engine_enabled": settings.EMBEDDING_ENGINE_ENABLED,
        "model_name": settings.EMBEDDING_MODEL_NAME,
        "model_status": status,
        "dimension": _MODEL_DIMENSION,
        "cache_enabled": settings.EMBEDDING_CACHE_ENABLED,
    }


def get_cache_stats() -> Dict[str, int]:
    """Returns current embedding cache hit/miss statistics."""
    return dict(_CACHE_STATS)


def clear_cache() -> None:
    """Clears embedding cache."""
    global _EMBEDDING_CACHE, _CACHE_STATS
    _EMBEDDING_CACHE.clear()
    _CACHE_STATS["cache_hits"] = 0
    _CACHE_STATS["cache_misses"] = 0
    _CACHE_STATS["embeddings_requested"] = 0
    _CACHE_STATS["embeddings_computed"] = 0
