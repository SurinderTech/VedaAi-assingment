"""
Student Feedback Service (Step 7).
Generates evidence-grounded feedback, strengths, weaknesses, and improvement recommendations.
STRICT SAFETY RULES:
- LLMs have ZERO authority over marks or score decisions.
- LLM response validation ignores any attempted score/mark fields.
- Falls back to local deterministic evidence text on any LLM failure.
"""
from __future__ import annotations
from typing import Dict, Any, List
from app.models.schemas import StructuredAssessmentResult, StructuredQuestionResult
from app.services.feedback_service import generate_question_evidence_feedback, generate_overall_assessment_feedback
from app.services.llm_provider import llm_complete_json
from app.core.config import settings


def generate_student_report_feedback(result: StructuredAssessmentResult) -> Dict[str, Any]:
    """
    Generates evidence-grounded assessment strengths, weaknesses, and improvement recommendations.
    Grounded strictly in actual Step 4/5 criteria and question outcomes.
    """
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommendations: List[str] = []

    for q in result.question_results:
        # Strengths from present criteria
        for c in q.criterion_results:
            if c.evidence_status == "present":
                strengths.append(f"Q{q.question_number}: Mastered '{c.description}'.")
            elif c.evidence_status in ("missing", "contradicted"):
                weaknesses.append(f"Q{q.question_number}: Missing concept '{c.description}'.")
                recommendations.append(f"Review '{c.description}' for Q{q.question_number} to strengthen core concept mastery.")
            elif c.evidence_status == "partially_present":
                weaknesses.append(f"Q{q.question_number}: Partial evidence for '{c.description}'.")
                recommendations.append(f"Elaborate further on '{c.description}' in Q{q.question_number}.")

        if q.status == "unanswered":
            weaknesses.append(f"Q{q.question_number} was unanswered.")
            recommendations.append(f"Ensure all questions are attempted, including Q{q.question_number}.")

    summary_text = (
        f"Overall Assessment Score: {result.final_awarded_marks}/{result.total_max_marks} ({result.percentage}%). "
        f"Answered {result.answered_questions} of {result.total_questions} questions."
    )

    return {
        "summary": summary_text,
        "strengths": strengths[:6],
        "weaknesses": weaknesses[:6],
        "recommendations": recommendations[:6],
    }


async def polish_student_report_with_llm(
    result: StructuredAssessmentResult,
    local_fb: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Polishes student feedback using LLM with strict validation.
    LLM output CANNOT modify marks, confidence, or grading decisions.
    Falls back to local feedback on any failure or malformed JSON.
    """
    if not settings.GRADING_LLM_ENABLED or not settings.PRIMARY_LLM_PROVIDER:
        return local_fb

    prompt = (
        f"Synthesize student-friendly assessment report summary and improvement guidance.\n"
        f"Final Score: {result.final_awarded_marks}/{result.total_max_marks} ({result.percentage}%)\n"
        f"Strengths: {local_fb['strengths']}\n"
        f"Weaknesses: {local_fb['weaknesses']}\n"
        f"Recommendations: {local_fb['recommendations']}\n"
        f"DO NOT assign or modify any scores or marks. Return JSON with keys 'summary', 'strengths', 'recommendations'."
    )
    try:
        res = await llm_complete_json(prompt, timeout=5.0)
        if isinstance(res, dict):
            # Ignore any attempted marks field in LLM response
            polished_summary = str(res.get("summary", local_fb["summary"])).strip()
            polished_strengths = res.get("strengths", local_fb["strengths"])
            if not isinstance(polished_strengths, list):
                polished_strengths = local_fb["strengths"]
            polished_recs = res.get("recommendations", local_fb["recommendations"])
            if not isinstance(polished_recs, list):
                polished_recs = local_fb["recommendations"]

            return {
                "summary": polished_summary,
                "strengths": [str(s) for s in polished_strengths[:6]],
                "weaknesses": local_fb["weaknesses"],
                "recommendations": [str(r) for r in polished_recs[:6]],
            }
    except Exception as e:
        print(f"[StudentFeedbackService] LLM report polish fallback: {e}")

    return local_fb
