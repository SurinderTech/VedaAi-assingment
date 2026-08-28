"""
Evidence-Grounded Feedback Service.
Generates structured feedback derived strictly from actual criterion evidence.
LLM may assist with natural language polishing, but MUST NEVER alter awarded marks,
criterion marks, or review state. Fallback to local evidence strings on LLM failure.
"""
from __future__ import annotations
from typing import Dict, Any, List
from app.models.schemas import StructuredQuestionResult, StructuredAssessmentResult
from app.services.llm_provider import llm_complete_json, LLMError
from app.core.config import settings


def generate_question_evidence_feedback(q: StructuredQuestionResult) -> Dict[str, Any]:
    """Generates deterministic local feedback grounded strictly in criterion evidence."""
    strengths: List[str] = []
    areas_to_improve: List[str] = []
    missing_concepts: List[str] = []
    corrections: List[str] = []

    for c in q.criterion_results:
        if c.evidence_status == "present":
            strengths.append(f"Demonstrated solid understanding of '{c.description}'.")
        elif c.evidence_status == "partially_present":
            areas_to_improve.append(f"Elaborate further on '{c.description}' to earn full marks.")
        elif c.evidence_status == "missing":
            missing_concepts.append(f"Missing required concept: '{c.description}'.")
        elif c.evidence_status == "contradicted":
            corrections.append(f"Incorrect/contradictory statement regarding '{c.description}'.")

    if not strengths and not areas_to_improve and not missing_concepts and not corrections:
        if q.status == "unanswered":
            missing_concepts.append("No response submitted for this question.")
        else:
            strengths.append("Answer provided.")

    feedback_text = (
        f"Score: {q.awarded_marks}/{q.max_marks}. " +
        (" Strengths: " + " ".join(strengths) if strengths else "") +
        (" Missing: " + " ".join(missing_concepts) if missing_concepts else "") +
        (" Improve: " + " ".join(areas_to_improve) if areas_to_improve else "") +
        (" Corrections: " + " ".join(corrections) if corrections else "")
    ).strip()

    return {
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "missing_concepts": missing_concepts,
        "corrections": corrections,
        "feedback_text": feedback_text,
    }


async def polish_feedback_with_llm(q: StructuredQuestionResult, local_fb: Dict[str, Any]) -> str:
    """
    Uses LLM to polish natural language feedback wording ONLY.
    Guarantees LLM call NEVER alters marks, criteria, or review status.
    Falls back gracefully to local feedback text on any LLM timeout/failure.
    """
    if not settings.GRADING_LLM_ENABLED or not settings.PRIMARY_LLM_PROVIDER:
        return local_fb["feedback_text"]

    prompt = (
        f"Synthesize concise teacher feedback for student answer.\n"
        f"Question: {q.question_text}\n"
        f"Marks: {q.awarded_marks}/{q.max_marks}\n"
        f"Strengths: {local_fb['strengths']}\n"
        f"Missing Concepts: {local_fb['missing_concepts']}\n"
        f"Areas to Improve: {local_fb['areas_to_improve']}\n"
        f"DO NOT assign scores or marks. Return JSON with key 'feedback'."
    )
    try:
        res = await llm_complete_json(prompt, timeout=5.0)
        if isinstance(res, dict) and res.get("feedback"):
            return str(res["feedback"]).strip()
    except Exception as e:
        print(f"[FeedbackService] LLM feedback polish notice: {e}")

    return local_fb["feedback_text"]


def generate_overall_assessment_feedback(result: StructuredAssessmentResult) -> Dict[str, Any]:
    """Generates overall assessment performance feedback summary from evidence."""
    total_q = result.total_questions
    final_score = result.final_awarded_marks
    total_max = result.total_max_marks
    pct = result.percentage

    strengths = []
    areas_for_improvement = []
    topics_requiring_attention = []

    for q in result.question_results:
        if q.awarded_marks >= q.max_marks * 0.8:
            strengths.append(f"Strong performance on Q{q.question_number} ({q.awarded_marks}/{q.max_marks})")
        elif q.awarded_marks <= q.max_marks * 0.4:
            areas_for_improvement.append(f"Needs improvement on Q{q.question_number} ({q.awarded_marks}/{q.max_marks})")
            for c in q.criterion_results:
                if c.evidence_status in ("missing", "contradicted"):
                    topics_requiring_attention.append(f"Q{q.question_number}: {c.description}")

    summary_text = (
        f"Assessment Overall Score: {final_score}/{total_max} ({pct}%). "
        f"{result.answered_questions}/{total_q} questions answered. "
        f"{result.questions_needing_review} questions pending teacher review."
    )

    return {
        "summary": summary_text,
        "percentage": pct,
        "strengths": strengths[:4],
        "areas_for_improvement": areas_for_improvement[:4],
        "topics_requiring_attention": topics_requiring_attention[:5],
        "questions_requiring_review": result.questions_needing_review,
    }
