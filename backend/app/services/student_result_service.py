"""
Student Result Service (Step 7).
Consumes finalized assessment results from Step 5/6 and formats student-facing performance data.
STRICT SAFETY RULES:
- MUST NOT re-grade answers or recalculate marks.
- MUST use final_awarded_marks from Step 5/6.
- Preserves 3-state separation (AI score, teacher score, final score).
- Performance bands are presentation-only display labels.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from app.models.schemas import (
    StructuredAssessmentResult,
    StructuredQuestionResult,
    StudentPerformanceSummary,
    QuestionPerformanceSummary,
    CriterionPerformanceSummary,
)
from app.services.feedback_service import generate_question_evidence_feedback


def determine_performance_band(percentage: float) -> str:
    """Returns presentation-only performance band label based strictly on score percentage."""
    if percentage >= 90.0:
        return "Outstanding"
    elif percentage >= 75.0:
        return "Proficient"
    elif percentage >= 50.0:
        return "Developing"
    else:
        return "Needs Support"


def build_student_performance_summary(
    result: StructuredAssessmentResult,
) -> StudentPerformanceSummary:
    """Builds overall student performance summary using authoritative final marks."""
    total_max = result.total_max_marks
    final_score = result.final_awarded_marks
    pct = round((final_score / total_max * 100.0), 2) if total_max > 0 else 0.0

    return StudentPerformanceSummary(
        assessment_id=result.assessment_id,
        total_max_marks=total_max,
        final_awarded_marks=final_score,
        percentage=pct,
        overall_confidence=result.overall_confidence,
        answered_questions=result.answered_questions,
        unanswered_questions=result.unanswered_questions,
        questions_needing_review=result.questions_needing_review,
        performance_band=determine_performance_band(pct),
    )


def build_question_performance_summary(
    q: StructuredQuestionResult,
) -> QuestionPerformanceSummary:
    """Builds individual question performance summary grounded in Step 4/5 evidence."""
    fb_dict = generate_question_evidence_feedback(q)
    pct = round((q.awarded_marks / q.max_marks * 100.0), 2) if q.max_marks > 0 else 0.0

    criteria_summaries: List[CriterionPerformanceSummary] = []
    for c in q.criterion_results:
        criteria_summaries.append(
            CriterionPerformanceSummary(
                criterion_id=c.criterion_id,
                description=c.description,
                max_marks=c.max_marks,
                awarded_marks=c.awarded_marks,
                evidence_status=c.evidence_status,
                confidence=c.confidence,
                provenance=c.provenance or "local",
            )
        )

    # Strengths: present criteria; Improvement: missing/partially_present/contradicted criteria
    strengths = fb_dict.get("strengths", [])
    improvements = fb_dict.get("areas_to_improve", []) + fb_dict.get("missing_concepts", []) + fb_dict.get("corrections", [])

    return QuestionPerformanceSummary(
        question_id=q.question_id,
        question_number=q.question_number,
        question_text=q.question_text,
        max_marks=q.max_marks,
        final_awarded_marks=q.awarded_marks,
        percentage=pct,
        status=q.status,
        feedback=q.feedback or fb_dict.get("feedback_text", ""),
        strengths=strengths,
        improvement_points=improvements,
        review_status=q.review_status,
        source_regions=q.answer_regions,
        criteria_summary=criteria_summaries,
    )
