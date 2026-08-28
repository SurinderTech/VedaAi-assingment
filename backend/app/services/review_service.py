"""
Teacher Review Queue Service.
Identifies and categorizes questions requiring teacher review based on actual evidence signals:
low_confidence, uncertain_evidence, contradiction, visual_uncertainty, math_uncertainty, llm_disagreement, mapping_uncertainty.
"""
from __future__ import annotations
from typing import List, Dict, Any
from app.models.schemas import StructuredQuestionResult, QuestionResult


def categorize_review_reason(q: StructuredQuestionResult) -> str:
    """Categorizes the primary escalation reason triggering teacher review."""
    if q.mapping_provenance == "review_required" or q.status == "review_required":
        return "mapping_uncertainty"
    if any(c.provenance == "conflict_flagged" for c in q.criterion_results):
        return "llm_disagreement"
    if any(c.evidence_status == "contradicted" for c in q.criterion_results):
        return "contradiction"
    if any(c.evidence_status == "uncertain" for c in q.criterion_results):
        return "uncertain_evidence"
    if q.evaluation_confidence < 0.55:
        return "low_confidence"
    if q.escalation_reason: # type: ignore
        return str(q.escalation_reason)
    return "teacher_requested_review"


def build_review_queue(structured_questions: List[StructuredQuestionResult]) -> Dict[str, Any]:
    """
    Constructs the structured teacher review queue containing all questions needing review.
    """
    pending_items = []
    reason_counts: Dict[str, int] = {}
    
    for q in structured_questions:
        if q.needs_review or q.review_status in ("PENDING_REVIEW", "TEACHER_OVERRIDE"):
            reason = categorize_review_reason(q)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            pending_items.append({
                "question_id": q.question_id,
                "question_number": q.question_number,
                "question_text": q.question_text,
                "max_marks": q.max_marks,
                "awarded_marks": q.awarded_marks,
                "confidence": q.evaluation_confidence,
                "review_reason": reason,
                "review_status": q.review_status,
                "has_teacher_override": q.teacher_adjusted_marks is not None,
                "answer_pages": q.answer_pages,
            })
            
    return {
        "pending_count": len(pending_items),
        "review_reasons_breakdown": reason_counts,
        "items": pending_items,
    }
