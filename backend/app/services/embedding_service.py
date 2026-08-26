"""
Semantic similarity between question text and answer text using SentenceTransformers.

Uses `all-MiniLM-L6-v2` for dense semantic embeddings.
Falls back gracefully to TF-IDF + cosine similarity if SentenceTransformers fails to load.
Does NOT use a vector database (in-memory similarity matrix per document).
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_model = None
_model_failed = False


def _get_transformer_model():
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[EmbeddingService] Failed to load SentenceTransformer ({e}). Falling back to TF-IDF.")
        _model_failed = True
        _model = None
    return _model


def similarity_matrix(questions: List[str], answers: List[str]) -> np.ndarray:
    """Returns an (len(questions) x len(answers)) matrix of similarity scores in [0, 1]."""
    if not questions or not answers:
        return np.zeros((len(questions), len(answers)))

    model = _get_transformer_model()
    if model is not None:
        try:
            q_embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
            a_embeddings = model.encode(answers, convert_to_numpy=True, normalize_embeddings=True)
            # Dot product of normalized vectors = cosine similarity
            sim = np.dot(q_embeddings, a_embeddings.T)
            return np.clip(sim, 0.0, 1.0)
        except Exception as e:
            print(f"[EmbeddingService] SentenceTransformer encoding failed ({e}), using TF-IDF fallback.")

    # Fallback: TF-IDF + Cosine Similarity
    corpus = questions + answers
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform(corpus)
        q_vecs = tfidf[: len(questions)]
        a_vecs = tfidf[len(questions):]
        sim = cosine_similarity(q_vecs, a_vecs)
        return np.clip(sim, 0.0, 1.0)
    except ValueError:
        return np.zeros((len(questions), len(answers)))
