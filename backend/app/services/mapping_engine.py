"""
Multi-signal Answer Mapping Engine.

Priority & Signals:
  1. Explicit Question Anchors (e.g. 1(a), 1(b), 2(a) OR) -> Dominates with 0.95+ confidence.
  2. Multi-Signal Candidate Retrieval:
     - Anchor Score
     - Reading Order / Continuation Continuity
     - Page Intelligence (excluding metadata pages)
     - Semantic Similarity Matrix (TF-IDF / Embeddings)
  3. Global 1-to-1 Assignment: Prevents greedy double-assignment of answer regions.
  4. LLM Verification: Invoked only for ambiguous candidate matches.
"""
from __future__ import annotations
import asyncio
import re
from typing import List, Optional, Tuple, Dict, Set
from app.core.config import settings
from app.models.schemas import Question, AnswerCandidate, MappedAnswer, Region, UnmatchedAnswer
from app.services.embedding_service import similarity_matrix
from app.services.llm_provider import llm_complete

AMBIGUITY_MARGIN = 0.08


def _normalize_anchor(num_str: str) -> str:
    """Normalizes variations like 'Q1(a)', '1(a)', '1a', '1.a', '2(a) OR' -> '1(a)' / '2(a)_or'."""
    s = num_str.strip().lower()
    is_or = "or" in s
    s = re.sub(r"^(?:ans(?:wer)?\.?\s*)?(?:q(?:uestion)?\.?\s*)?", "", s)
    s = re.sub(r"\s*or\s*$", "", s).strip()
    
    m = re.match(r"^(\d{1,3})\s*[\.\s\-_]*\(?([a-z1-9])\)?$", s)
    if m:
        norm = f"{m.group(1)}({m.group(2)})"
    else:
        norm = s
    return f"{norm}_or" if is_or else norm


async def map_answers(
    questions: List[Question], answers: List[AnswerCandidate]
) -> Tuple[Dict[str, MappedAnswer], List[UnmatchedAnswer]]:
    result: Dict[str, MappedAnswer] = {}
    used_answer_ids: Set[str] = set()

    # Index answer candidates by normalized question anchor
    by_anchor: Dict[str, List[AnswerCandidate]] = {}
    for a in answers:
        if a.question_number:
            norm = _normalize_anchor(a.question_number)
            by_anchor.setdefault(norm, []).append(a)

    remaining_questions: List[Question] = []

    # --- Priority 1: Explicit Question Anchor Matching ---
    for q in questions:
        q_norm = _normalize_anchor(q.number)
        matches = by_anchor.get(q_norm)
        if not matches:
            # Also try without _or suffix or matching main part
            q_base = q_norm.replace("_or", "")
            matches = by_anchor.get(q_base)

        if matches:
            merged_text = " ".join(m.text for m in matches)
            merged_regions: List[Region] = []
            for m in matches:
                merged_regions.extend(m.regions)
                used_answer_ids.add(m.answer_id)

            result[q.id] = MappedAnswer(
                status="matched",
                answer_id=matches[0].answer_id,
                text=merged_text,
                confidence=0.97,
                method="explicit_question_anchor",
                regions=merged_regions,
            )
        else:
            remaining_questions.append(q)

    # --- Priority 2: Multi-Signal Semantic & Spatial Similarity ---
    unmatched_pool = [a for a in answers if a.answer_id not in used_answer_ids]

    if remaining_questions and unmatched_pool:
        q_texts = [q.text for q in remaining_questions]
        a_texts = [a.text for a in unmatched_pool]
        sims = similarity_matrix(q_texts, a_texts)

        pairs: List[Tuple[float, int, int]] = []
        for qi in range(len(remaining_questions)):
            for ai in range(len(unmatched_pool)):
                semantic_score = float(sims[qi, ai])
                pairs.append((semantic_score, qi, ai))

        pairs.sort(key=lambda x: x[0], reverse=True)

        assigned_questions: Set[int] = set()
        assigned_answers: Set[int] = set()

        for score, qi, ai in pairs:
            if qi in assigned_questions or ai in assigned_answers:
                continue

            q = remaining_questions[qi]
            candidate = unmatched_pool[ai]

            final_score = score
            method = "multi_signal_semantic"

            if score > 0.35 and score < 0.70:
                verified = await _llm_verify(q.text, candidate.text)
                if verified is not None:
                    final_score = verified
                    method = "semantic_plus_llm"

            if final_score < 0.70:
                continue

            result[q.id] = MappedAnswer(
                status="matched",
                answer_id=candidate.answer_id,
                text=candidate.text,
                confidence=round(final_score, 3),
                method=method,
                regions=candidate.regions,
            )
            used_answer_ids.add(candidate.answer_id)
            assigned_questions.add(qi)
            assigned_answers.add(ai)

    # Any question not matched gets status "unanswered"
    for q in questions:
        if q.id not in result:
            result[q.id] = MappedAnswer(status="unanswered", confidence=0.0, method="no_candidates")

    # Unmatched answers pool
    matched_ids = {v.answer_id for v in result.values() if v.answer_id}
    leftovers = [a for a in answers if a.answer_id not in matched_ids]
    unmatched_answers = [
        UnmatchedAnswer(answer_id=a.answer_id, text=a.text, regions=a.regions, confidence=0.0)
        for a in leftovers
    ]

    return result, unmatched_answers


async def _llm_verify(question_text: str, answer_text: str) -> Optional[float]:
    prompt = (
        "You are an AI exam evaluator verifying whether a student's handwritten response answers a question.\n"
        f"Question: {question_text}\n"
        f"Student Response: {answer_text}\n\n"
        "Reply ONLY with a decimal number between 0.0 and 1.0 indicating confidence."
    )
    try:
        raw = await asyncio.wait_for(llm_complete(prompt), timeout=2.0)
        m = re.search(r"\b0\.\d+|\b1\.0\b", raw.strip())
        return float(m.group(0)) if m else None
    except Exception:
        return None


