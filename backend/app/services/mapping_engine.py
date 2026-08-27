"""
Multi-signal answer mapping (plan sections 19-24).

Priority order:
  1. Explicit question number written by the student -> very high confidence.
  2. One-to-one semantic similarity matching using SentenceTransformers / TF-IDF.
  3. LLM verification, invoked only when top candidates are genuinely ambiguous.

Guarantees 1-to-1 mapping: once an answer candidate is assigned to a question,
it cannot be assigned to another question.
"""
from __future__ import annotations
import asyncio
from typing import List, Optional, Tuple
from app.core.config import settings
from app.models.schemas import Question, AnswerCandidate, MappedAnswer, Region, UnmatchedAnswer
from app.services.embedding_service import similarity_matrix
from app.services.llm_provider import llm_complete, LLMError

AMBIGUITY_MARGIN = 0.08  # ask LLM if top 2 candidates are within this margin


def _status_for_score(score: float) -> str:
    if score >= settings.HIGH_CONFIDENCE:
        return "matched"
    if score >= settings.MEDIUM_CONFIDENCE:
        return "review_required"
    return "unmatched"


async def map_answers(
    questions: List[Question], answers: List[AnswerCandidate]
) -> Tuple[dict[str, MappedAnswer], List[UnmatchedAnswer]]:
    result: dict[str, MappedAnswer] = {}
    used_answer_ids: set[str] = set()

    # --- Priority 1: Explicit Question Number written by student ---
    by_number: dict[str, List[AnswerCandidate]] = {}
    for a in answers:
        if a.question_number:
            # Normalize e.g. "11a" -> "11(a)" if needed
            clean_num = a.question_number.strip().lower()
            by_number.setdefault(clean_num, []).append(a)

    remaining_questions: List[Question] = []
    for q in questions:
        q_num = q.number.strip().lower()
        matches = by_number.get(q_num)
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
                method="explicit_question_number",
                regions=merged_regions,
            )
        else:
            remaining_questions.append(q)

    # --- Priority 2: One-to-One Semantic Similarity ---
    unmatched_pool = [a for a in answers if a.answer_id not in used_answer_ids]

    if remaining_questions and unmatched_pool:
        q_texts = [q.text for q in remaining_questions]
        a_texts = [a.text for a in unmatched_pool]
        sims = similarity_matrix(q_texts, a_texts)

        # Collect all (score, q_idx, a_idx) pairs and sort descending
        pairs: List[Tuple[float, int, int]] = []
        for qi in range(len(remaining_questions)):
            for ai in range(len(unmatched_pool)):
                pairs.append((float(sims[qi, ai]), qi, ai))

        pairs.sort(key=lambda x: x[0], reverse=True)

        assigned_questions: set[int] = set()
        assigned_answers: set[int] = set()

        for score, qi, ai in pairs:
            if qi in assigned_questions or ai in assigned_answers:
                continue

            q = remaining_questions[qi]
            candidate = unmatched_pool[ai]

            # Find second best score for this question to check for ambiguity
            other_scores = [sims[qi, k] for k in range(len(unmatched_pool)) if k != ai]
            second_score = max(other_scores) if other_scores else 0.0

            final_score = score
            method = "semantic_similarity"

            # --- Priority 3: LLM verification if top candidates are ambiguous ---
            if score > 0.3 and (score - second_score) < AMBIGUITY_MARGIN and score < settings.HIGH_CONFIDENCE:
                verified = await _llm_verify(q.text, candidate.text)
                if verified is not None:
                    final_score = verified
                    method = "semantic_plus_llm"

            if score <= 0.05:
                # Score too low to match
                continue

            status = _status_for_score(final_score)
            if status == "unmatched":
                continue

            # Assign 1-to-1 match
            result[q.id] = MappedAnswer(
                status=status,
                answer_id=candidate.answer_id,
                text=candidate.text,
                confidence=round(final_score, 3),
                method=method,
                regions=candidate.regions,
            )
            used_answer_ids.add(candidate.answer_id)
            assigned_questions.add(qi)
            assigned_answers.add(ai)

    # Any remaining question never matched gets status "unanswered"
    for q in questions:
        if q.id not in result:
            result[q.id] = MappedAnswer(status="unanswered", confidence=0.0, method="no_candidates")

    # --- Unmatched Answers Pool ---
    matched_ids = {v.answer_id for v in result.values() if v.answer_id}
    leftovers = [a for a in answers if a.answer_id not in matched_ids]
    unmatched_answers = [
        UnmatchedAnswer(answer_id=a.answer_id, text=a.text, regions=a.regions, confidence=0.0)
        for a in leftovers
    ]

    return result, unmatched_answers


async def _llm_verify(question_text: str, answer_text: str) -> Optional[float]:
    prompt = (
        "You are verifying whether a student's answer responds to a given exam question.\n"
        f"Question: {question_text}\n"
        f"Student answer: {answer_text}\n"
        "On a scale of 0.0 to 1.0, how likely is it that this answer responds to this "
        "question? Reply with ONLY a decimal number, nothing else."
    )
    try:
        raw = await asyncio.wait_for(llm_complete(prompt), timeout=2.0)
        return max(0.0, min(1.0, float(raw.strip().split()[0])))
    except Exception:
        return None

