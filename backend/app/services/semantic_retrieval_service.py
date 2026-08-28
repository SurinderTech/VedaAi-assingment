"""
Step 8 — Semantic Candidate Retrieval Service.
Ranks candidate answer regions for each question by dense embedding similarity,
calculates candidate ambiguity margin (best_similarity - second_best_similarity),
and supplies top-K candidate sets to Step 3 Mapping Engine without assigning final mapping.
"""
from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.models.schemas import Question, AnswerRegion
from app.services.embedding_service import (
    embed_texts,
    cosine_similarity_matrix,
    get_model_metadata,
    similarity_matrix_tfidf,
)


def get_semantic_candidates(
    questions: List[Question],
    answer_regions: List[AnswerRegion],
    top_k: int = settings.EMBEDDING_TOP_K,
) -> Dict[str, Any]:
    """
    Ranks top-K candidate answer regions for each question based on semantic similarity.
    Calculates ambiguity margins and provides performance metrics.
    STRICT PROTECTION: Does NOT assign final question->answer mapping.
    """
    start_time = time.time()
    total_questions = len(questions)

    if not questions or not answer_regions:
        return {
            "status": "unavailable",
            "question_candidates": {},
            "similarity_matrix": np.zeros((len(questions), len(answer_regions))),
            "metrics": {
                "total_questions": total_questions,
                "embedding_evaluations": 0,
                "clear_semantic_matches": 0,
                "ambiguous_semantic_matches": 0,
                "llm_calls_avoided": 0,
                "embedding_failures": 0,
                "retrieval_time_ms": 0.0,
            },
        }

    q_texts = [q.text for q in questions]
    a_texts = [a.text for a in answer_regions]

    meta = get_model_metadata()
    model_status = meta.get("model_status", "UNAVAILABLE")

    # Try embedding-based vector retrieval
    sim_matrix = None
    embedding_failed = False

    if settings.EMBEDDING_ENGINE_ENABLED and model_status == "AVAILABLE":
        try:
            q_vecs = embed_texts(q_texts)
            a_vecs = embed_texts(a_texts)

            if q_vecs is not None and a_vecs is not None and len(q_vecs) > 0 and len(a_vecs) > 0:
                sim_matrix = cosine_similarity_matrix(q_vecs, a_vecs)
            else:
                embedding_failed = True
        except Exception as e:
            print(f"[SemanticRetrieval] Embedding computation exception: {e}")
            embedding_failed = True
    else:
        embedding_failed = True

    # Fallback to TF-IDF if embeddings are unavailable or failed
    if sim_matrix is None or embedding_failed:
        sim_matrix = similarity_matrix_tfidf(q_texts, a_texts)
        retrieval_mode = "tfidf_fallback"
    else:
        retrieval_mode = "dense_embeddings"

    question_candidates: Dict[str, Dict[str, Any]] = {}
    clear_matches = 0
    ambiguous_matches = 0
    llm_avoided = 0

    margin_threshold = settings.SEMANTIC_AMBIGUITY_MARGIN

    for q_idx, q in enumerate(questions):
        scores = sim_matrix[q_idx] if q_idx < len(sim_matrix) else np.zeros(len(answer_regions))
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        candidates_list = []
        for rank, idx in enumerate(ranked_indices):
            region = answer_regions[idx]
            sim_score = float(scores[idx])
            candidates_list.append({
                "answer_id": region.answer_id,
                "question_anchor": region.question_anchor,
                "text_snippet": region.text[:100] if region.text else "",
                "semantic_similarity": round(sim_score, 4),
                "rank": rank + 1,
                "page": region.pages[0] if region.pages else 1,
            })

        best_score = float(scores[ranked_indices[0]]) if len(ranked_indices) > 0 else 0.0
        second_best_score = float(scores[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
        margin = round(best_score - second_best_score, 4)

        if best_score >= 0.60 and margin >= margin_threshold:
            sem_status = "clear"
            clear_matches += 1
            llm_avoided += 1
        elif best_score >= 0.35 and margin < margin_threshold:
            sem_status = "ambiguous"
            ambiguous_matches += 1
        else:
            sem_status = "low_similarity"

        question_candidates[q.id] = {
            "question_id": q.id,
            "question_number": q.number,
            "top_candidates": candidates_list,
            "best_similarity": round(best_score, 4),
            "second_best_similarity": round(second_best_score, 4),
            "margin": margin,
            "semantic_status": sem_status,
        }

    retrieval_time = round((time.time() - start_time) * 1000.0, 2)

    return {
        "status": "completed",
        "retrieval_mode": retrieval_mode,
        "question_candidates": question_candidates,
        "similarity_matrix": sim_matrix,
        "metrics": {
            "total_questions": total_questions,
            "embedding_evaluations": len(q_texts) + len(a_texts),
            "clear_semantic_matches": clear_matches,
            "ambiguous_semantic_matches": ambiguous_matches,
            "llm_calls_avoided": llm_avoided,
            "embedding_failures": 1 if embedding_failed else 0,
            "retrieval_time_ms": retrieval_time,
        },
    }
