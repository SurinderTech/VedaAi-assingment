"""
Step 9 — Assessment Intelligence & Teacher Insights Service.

Lightweight, evidence-grounded assessment insights service.
STRICT SAFEGUARDS:
- Uses final_awarded_marks as authoritative score.
- NEVER recalculates marks, modifies grading decisions, or alters review state.
- Grounded strictly in actual Step 4/5 criterion evidence.
- Preserves AI/Teacher separation and exact BBoxes.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.models.schemas import (
    StructuredAssessmentResult,
    StructuredQuestionResult,
    AssessmentInsight,
    QuestionInsight,
    AssessmentInsights,
)
from app.services.llm_provider import llm_complete_json
from app.core.config import settings


def generate_assessment_insights(
    result: StructuredAssessmentResult,
) -> AssessmentInsights:
    """
    Generates concise, evidence-grounded teacher insights for an assessment.
    Consumes finalized Step 5 result without altering scores or grading state.
    """
    assessment_id = result.assessment_id
    final_score = result.final_awarded_marks
    total_max = result.total_max_marks
    pct = result.percentage

    answered_count = result.answered_questions
    unanswered_count = result.unanswered_questions
    unmatched_count = result.unmatched_answers_count
    review_count = result.questions_needing_review

    overall_strengths: List[str] = []
    overall_attention: List[str] = []

    question_insights: List[QuestionInsight] = []
    review_priorities: List[AssessmentInsight] = []
    error_patterns: List[AssessmentInsight] = []

    # Map of error pattern categories
    pattern_buckets: Dict[str, List[Dict[str, Any]]] = {
        "CONCEPTUAL_GAP": [],
        "PARTIAL_CONCEPT": [],
        "MISCONCEPTION": [],
        "INCOMPLETE_RESPONSE": [],
        "UNANSWERED_PATTERN": [],
        "CALCULATION_OR_PROCEDURAL_ERROR": [],
    }

    for q in result.question_results:
        q_strengths: List[str] = []
        q_improvements: List[str] = []
        q_patterns: List[str] = []
        q_evidence_refs: List[str] = []

        # Unanswered Question Check
        if q.status == "unanswered":
            msg = f"Q{q.question_number} was left unanswered."
            q_improvements.append(msg)
            pattern_buckets["UNANSWERED_PATTERN"].append({
                "q_id": q.question_id,
                "q_num": q.question_number,
                "ref": f"q:{q.question_id}:unanswered",
                "desc": msg,
            })
            question_insights.append(
                QuestionInsight(
                    question_id=q.question_id,
                    question_number=q.question_number,
                    strengths=[],
                    improvement_areas=q_improvements,
                    error_patterns=["UNANSWERED_PATTERN"],
                    evidence_refs=[f"q:{q.question_id}:unanswered"],
                    source_regions=q.answer_regions,
                    confidence=1.0,
                )
            )
            continue

        # Criteria Inspection
        for c in q.criterion_results:
            c_ref = f"q:{q.question_id}:c:{c.criterion_id}"
            q_evidence_refs.append(c_ref)

            if c.evidence_status == "present":
                s_msg = f"Q{q.question_number}: Correctly demonstrated '{c.description}'."
                q_strengths.append(s_msg)
                if len(overall_strengths) < 6:
                    overall_strengths.append(s_msg)

            elif c.evidence_status == "partially_present":
                imp_msg = f"Q{q.question_number}: Partial evidence for '{c.description}'."
                q_improvements.append(imp_msg)
                pattern_buckets["PARTIAL_CONCEPT"].append({
                    "q_id": q.question_id,
                    "q_num": q.question_number,
                    "ref": c_ref,
                    "desc": f"Partial understanding of '{c.description}'",
                })

            elif c.evidence_status in ("missing", "contradicted"):
                imp_msg = f"Q{q.question_number}: Missing required concept '{c.description}'."
                q_improvements.append(imp_msg)
                overall_attention.append(imp_msg)

                p_type = "MISCONCEPTION" if c.evidence_status == "contradicted" else "CONCEPTUAL_GAP"
                pattern_buckets[p_type].append({
                    "q_id": q.question_id,
                    "q_num": q.question_number,
                    "ref": c_ref,
                    "desc": f"Missing/contradicted '{c.description}'",
                })

            elif c.evidence_status == "uncertain":
                unc_msg = f"Q{q.question_number}: Evidence for '{c.description}' is inconclusive and may require review."
                q_improvements.append(unc_msg)

        # Review Priority Detection
        if q.review_status != "NOT_REQUIRED" or q.needs_review:
            p_title = f"Review Question {q.question_number}"
            p_summary = f"Q{q.question_number} is flagged for teacher review (Status: {q.review_status}, Confidence: {round(q.evaluation_confidence * 100)}%)."
            review_priorities.append(
                AssessmentInsight(
                    insight_id=f"prio_{q.question_id}",
                    type="REVIEW_PRIORITY",
                    title=p_title,
                    summary=p_summary,
                    question_ids=[q.question_id],
                    evidence_refs=q_evidence_refs,
                    confidence=q.evaluation_confidence,
                    source="review_service",
                )
            )

        question_insights.append(
            QuestionInsight(
                question_id=q.question_id,
                question_number=q.question_number,
                strengths=q_strengths,
                improvement_areas=q_improvements,
                error_patterns=q_patterns,
                evidence_refs=q_evidence_refs,
                source_regions=q.answer_regions,
                confidence=q.evaluation_confidence,
            )
        )

    # Process Error Patterns with "Observed issue" vs "Recurring pattern observed"
    for p_type, Occurrences in pattern_buckets.items():
        if not Occurrences:
            continue

        q_ids = list(dict.fromkeys(item["q_num"] for item in Occurrences))
        raw_q_ids = list(dict.fromkeys(item["q_id"] for item in Occurrences))
        refs = [item["ref"] for item in Occurrences]
        count = len(Occurrences)

        if count >= 2:
            title = f"Recurring pattern observed: {p_type.replace('_', ' ').title()}"
            summary = f"Identified across {count} questions (Q{', Q'.join(q_ids)})."
        else:
            title = f"Observed issue: {p_type.replace('_', ' ').title()}"
            summary = f"Observed in Q{q_ids[0]}."

        error_patterns.append(
            AssessmentInsight(
                insight_id=f"err_{p_type.lower()}",
                type="ERROR_PATTERN",
                title=title,
                summary=summary,
                question_ids=raw_q_ids,
                evidence_refs=refs,
                confidence=0.9,
                source="error_service",
            )
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    return AssessmentInsights(
        assessment_id=assessment_id,
        final_awarded_marks=final_score,
        total_max_marks=total_max,
        percentage=pct,
        answered_questions=answered_count,
        unanswered_questions=unanswered_count,
        unmatched_answers_count=unmatched_count,
        questions_needing_review=review_count,
        strengths=overall_strengths[:6],
        areas_needing_attention=overall_attention[:6],
        error_patterns=error_patterns,
        review_priorities=review_priorities,
        question_insights=question_insights,
        generated_at=now_iso,
    )


async def polish_insights_with_llm(
    insights: AssessmentInsights,
) -> AssessmentInsights:
    """
    Polishes teacher insight summary with LLM with ZERO mark authority.
    Falls back gracefully to deterministic insights on LLM error/timeout.
    """
    if not settings.GRADING_LLM_ENABLED or not settings.PRIMARY_LLM_PROVIDER:
        return insights

    prompt = (
        f"Synthesize concise teacher-facing assessment insights.\n"
        f"Score: {insights.final_awarded_marks}/{insights.total_max_marks} ({insights.percentage}%)\n"
        f"Strengths: {insights.strengths}\n"
        f"Attention Needed: {insights.areas_needing_attention}\n"
        f"DO NOT change scores, marks, or grading decisions. Return JSON with 'strengths' and 'areas_needing_attention'."
    )
    try:
        res = await llm_complete_json(prompt, timeout=5.0)
        if isinstance(res, dict):
            polished_s = res.get("strengths", insights.strengths)
            polished_a = res.get("areas_needing_attention", insights.areas_needing_attention)
            if isinstance(polished_s, list):
                insights.strengths = [str(s) for s in polished_s[:6]]
            if isinstance(polished_a, list):
                insights.areas_needing_attention = [str(a) for a in polished_a[:6]]
    except Exception as e:
        print(f"[AssessmentInsightService] LLM polish fallback: {e}")

    return insights
