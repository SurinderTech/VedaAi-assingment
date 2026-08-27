"""
Semantic similarity between question text and answer text using TF-IDF + Cosine Similarity.

Provides fast, local, zero-network similarity computation (0ms overhead) with zero HF Hub rate limit warnings.
Does NOT use a vector database (in-memory similarity matrix per document).
"""
from __future__ import annotations
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def similarity_matrix(questions: List[str], answers: List[str]) -> np.ndarray:
    """Returns an (len(questions) x len(answers)) matrix of similarity scores in [0, 1]."""
    if not questions or not answers:
        return np.zeros((len(questions), len(answers)))

    # Fallback: TF-IDF + Cosine Similarity
    corpus = questions + answers
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform(corpus)
        q_vecs = tfidf[: len(questions)]
        a_vecs = tfidf[len(questions):]
        sim = cosine_similarity(q_vecs, a_vecs)
        return np.clip(sim, 0.0, 1.0)
    except Exception:
        return np.zeros((len(questions), len(answers)))

