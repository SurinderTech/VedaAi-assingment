"""
Assessment Result Service.
Master aggregator for Step 5: Assessment Results, Teacher Overrides, Mark Validation, and Finalization.
Preserves 3 distinct states: AI Evaluation, Teacher Review, Final Result.
Never mutates finalized results silently—creates explicit version revisions.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from app.models.schemas import (
    AssessmentResult, QuestionResult, UnmatchedAnswer,
    StructuredAssessmentResult, StructuredQuestionResult, CriterionResult,
    TeacherReview, AuditEvent, AssessmentRevision, ReviewStatus
)
from app.services.explanation_service import build_question_explanation
from app.services.review_service import build_review_queue
from app.services.feedback_service import generate_question_evidence_feedback, generate_overall_assessment_feedback
from app.services.audit_service import create_audit_event
from app.core import store


def build_structured_assessment_result(
    assessment_id: str,
    question_results: List[QuestionResult],
    unmatched_answers: List[UnmatchedAnswer],
) -> StructuredAssessmentResult:
    """
    Consumes real Step 1-4 outputs and constructs the unified, explainable assessment result.
    Preserves exact Step 4 provenance, criteria, confidence, and Step 2 BBoxes.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    structured_qs: List[StructuredQuestionResult] = []
    
    total_q = len(question_results)
    answered_q = 0
    unanswered_q = 0
    total_max = 0.0
    ai_awarded = 0.0
    
    for q_res in question_results:
        g_details = q_res.grading.result_details if (q_res.grading and q_res.grading.result_details) else None
        mapped_ans = q_res.answer
        
        max_m = g_details.max_marks if g_details else (q_res.grading.max_score if (q_res.grading and q_res.grading.max_score) else 2.0)
        ai_marks = g_details.awarded_marks if g_details else (q_res.grading.score if (q_res.grading and q_res.grading.score) else 0.0)
        conf = g_details.confidence if g_details else 0.50
        needs_rev = g_details.needs_review if g_details else False
        eval_method = g_details.evaluation_method if g_details else "local"
        map_prov = mapped_ans.status if mapped_ans else "unmatched"
        
        # Check Answer Status
        ans_status = mapped_ans.status if mapped_ans else "unanswered"
        has_content = bool(mapped_ans and ((mapped_ans.text and mapped_ans.text.strip()) or mapped_ans.regions))
        if ans_status == "unanswered" or not has_content:
            unanswered_q += 1
            ans_status = "unanswered"
        else:
            answered_q += 1
            
        total_max += max_m
        ai_awarded += ai_marks
        
        # Extract BBoxes from Step 2 geometry
        answer_pages = []
        source_regions = []
        if mapped_ans and mapped_ans.regions:
            for reg in mapped_ans.regions:
                if reg.page not in answer_pages:
                    answer_pages.append(reg.page)
                source_regions.append({
                    "page": reg.page,
                    "bbox": {
                        "x": reg.bbox.x,
                        "y": reg.bbox.y,
                        "width": reg.bbox.width,
                        "height": reg.bbox.height,
                    },
                    "text_snippet": reg.text_content[:100] if hasattr(reg, "text_content") and reg.text_content else "",
                })
                
        # Build Criterion Results using REAL Step 4 evidence and provenance
        criterion_results: List[CriterionResult] = []
        raw_criteria = g_details.criteria if g_details else []
        for ev in raw_criteria:
            criterion_results.append(
                CriterionResult(
                    criterion_id=ev.criterion_id,
                    description=ev.description,
                    max_marks=ev.max_marks,
                    awarded_marks=ev.awarded_marks,
                    evidence_status=ev.status,
                    evidence_text=ev.evidence_text,
                    confidence=ev.confidence,
                    provenance=ev.provenance or "local",
                    needs_review=(ev.status == "uncertain" or ev.confidence < 0.55),
                    source_regions=source_regions,
                )
            )
            
        rev_status: ReviewStatus = "PENDING_REVIEW" if needs_rev else "NOT_REQUIRED"
        
        # Fix 5: Build structured options from extracted_options if available
        structured_opts: list = []
        if hasattr(q_res, "extracted_options") and q_res.extracted_options:
            for eo in q_res.extracted_options:
                structured_opts.append({
                    "option_id": getattr(eo, "option_id", ""),
                    "label": getattr(eo, "label", ""),
                    "text": getattr(eo, "text", ""),
                    "full_text": getattr(eo, "text", ""),
                    "source_region_ids": getattr(eo, "source_region_ids", []),
                    "confidence": getattr(eo, "extraction_confidence", 1.0),
                })
        
        # Build Local Evidence Feedback
        struct_q_pre = StructuredQuestionResult(
            question_id=q_res.id,
            question_number=q_res.number,
            question_text=q_res.text,
            max_marks=max_m,
            answer_id=mapped_ans.answer_id if mapped_ans else None,
            answer_text=(mapped_ans.text or "").strip() if mapped_ans else "",
            answer_pages=answer_pages,
            answer_regions=source_regions,
            status=ans_status, # type: ignore
            awarded_marks=ai_marks,
            original_ai_marks=ai_marks,
            teacher_adjusted_marks=None,
            evaluation_confidence=conf,
            needs_review=needs_rev,
            criterion_results=criterion_results,
            evidence_summary=g_details.correct_evidence + g_details.missing_evidence if g_details else [],
            feedback="",
            mapping_provenance=map_prov,
            grading_provenance=eval_method,
            escalation_reason=g_details.escalation_reason if g_details else None,
            review_status=rev_status,
            # Preserve VLM-extracted MCQ options end-to-end
            options=q_res.options if q_res.options else [],
            # Fix 5: Full semantic structure preserved through API
            question_type=getattr(q_res, "question_type", "UNKNOWN") or "UNKNOWN",
            parent_question_id=getattr(q_res, "parent_question_id", None),
            page_number=getattr(q_res, "page", 0),
            semantic_state=getattr(q_res, "verification_state", "UNKNOWN") or "UNKNOWN",
            source_region_ids=getattr(q_res, "source_region_ids", []) or [],
            extraction_confidence=getattr(q_res, "extraction_confidence", 1.0),
            extracted_options=structured_opts,
            correct_option=getattr(q_res, "correct_option", None),
            correct_answer=getattr(q_res, "correct_answer", None),
        )
        
        fb_dict = generate_question_evidence_feedback(struct_q_pre)
        struct_q_pre.feedback = fb_dict["feedback_text"]
        structured_qs.append(struct_q_pre)
        
    final_awarded = ai_awarded
    pct = round((final_awarded / total_max) * 100, 2) if total_max > 0 else 0.0
    overall_conf = round(sum(q.evaluation_confidence for q in structured_qs) / max(1, total_q), 2)
    review_queue = build_review_queue(structured_qs)
    
    init_audit = create_audit_event(
        assessment_id=assessment_id,
        event_type="AI_GRADING_COMPLETED",
        source="system",
        reason="Initial AI evaluation pipeline execution completed.",
        new_value={"total_questions": total_q, "ai_awarded_marks": ai_awarded, "percentage": pct},
    )
    
    res = StructuredAssessmentResult(
        assessment_id=assessment_id,
        assessment_status="IN_REVIEW",
        revision_index=1,
        total_questions=total_q,
        answered_questions=answered_q,
        unanswered_questions=unanswered_q,
        unmatched_answers_count=len(unmatched_answers),
        total_max_marks=total_max,
        ai_awarded_marks=ai_awarded,
        teacher_adjusted_marks=None,
        final_awarded_marks=final_awarded,
        percentage=pct,
        overall_confidence=overall_conf,
        questions_needing_review=review_queue["pending_count"],
        question_results=structured_qs,
        review_summary=review_queue,
        grading_statistics={
            "local_only_questions": sum(1 for q in structured_qs if q.grading_provenance == "local"),
            "llm_assisted_questions": sum(1 for q in structured_qs if q.grading_provenance in ("local+llm", "llm")),
            "fallback_questions": sum(1 for q in structured_qs if q.grading_provenance == "local_fallback"),
        },
        audit_trail=[init_audit],
        version_history=[],
        created_at=now_iso,
        updated_at=now_iso,
    )
    return res


def validate_teacher_override_marks(q: StructuredQuestionResult, teacher_marks: float, criterion_overrides: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
    """Validates that teacher marks satisfy 0 <= awarded <= max_marks and criteria sum = question marks."""
    if teacher_marks < 0.0 or teacher_marks > q.max_marks:
        return False, f"Teacher marks {teacher_marks} out of range [0.0, {q.max_marks}] for Q{q.question_number}"
        
    if criterion_overrides:
        c_sum = round(sum(criterion_overrides.values()), 2)
        if abs(c_sum - teacher_marks) > 0.05:
            return False, f"Sum of criterion marks ({c_sum}) does not match question override marks ({teacher_marks}) for Q{q.question_number}"
            
    return True, "Valid"


def apply_teacher_override(
    assessment_res: AssessmentResult,
    question_id: str,
    teacher_marks: float,
    criterion_overrides: Optional[Dict[str, float]] = None,
    comment: Optional[str] = None,
    reason: str = "Teacher manual override",
    reviewer: str = "Teacher",
) -> StructuredAssessmentResult:
    """
    Applies additive teacher override while preserving original AI decision.
    If assessment is FINALIZED, creates a new explicit revision rather than mutating in-place.
    Generates immutable audit trail events for every change.
    """
    struct_res = assessment_res.structured_result
    if not struct_res:
        raise ValueError("Assessment structured result not found")
        
    struct_res.assessment_id = assessment_res.assessment_id
    target_q = next((q for q in struct_res.question_results if q.question_id == question_id), None)
    if not target_q:
        raise ValueError(f"Question '{question_id}' not found in assessment")
        
    # Validate marks
    valid, msg = validate_teacher_override_marks(target_q, teacher_marks, criterion_overrides)
    if not valid:
        raise ValueError(msg)
        
    now_iso = datetime.now(timezone.utc).isoformat()
    prev_q_marks = target_q.awarded_marks
    prev_review_status = target_q.review_status
    
    # Check if assessment is FINALIZED -> create new revision version!
    was_finalized = (struct_res.assessment_status == "FINALIZED")
    if was_finalized:
        struct_res.revision_index += 1
        
    # Apply Criterion Overrides if provided
    if criterion_overrides:
        for c in target_q.criterion_results:
            if c.criterion_id in criterion_overrides:
                c_prev = c.awarded_marks
                c.awarded_marks = criterion_overrides[c.criterion_id]
                c.evidence_status = "present" if c.awarded_marks >= c.max_marks else ("partially_present" if c.awarded_marks > 0 else "missing")
                # Audit criterion override
                struct_res.audit_trail.append(
                    create_audit_event(
                        assessment_id=struct_res.assessment_id,
                        question_id=question_id,
                        event_type="TEACHER_CRITERION_OVERRIDE",
                        previous_value={"criterion_id": c.criterion_id, "marks": c_prev},
                        new_value={"criterion_id": c.criterion_id, "marks": c.awarded_marks},
                        source=reviewer,
                        reason=reason,
                    )
                )

    # Apply Question Overrides (Preserving original_ai_marks!)
    target_q.teacher_adjusted_marks = teacher_marks
    target_q.awarded_marks = teacher_marks
    target_q.review_status = "TEACHER_OVERRIDE"
    target_q.needs_review = False
    
    t_review = TeacherReview(
        review_id=f"rev_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        question_id=question_id,
        original_ai_marks=target_q.original_ai_marks,
        teacher_marks=teacher_marks,
        reason=reason,
        comment=comment,
        reviewer=reviewer,
        timestamp=now_iso,
        changed=True,
    )
    target_q.teacher_review = t_review
    
    # Audit Question Mark Override
    struct_res.audit_trail.append(
        create_audit_event(
            assessment_id=struct_res.assessment_id,
            question_id=question_id,
            event_type="TEACHER_MARK_OVERRIDE",
            previous_value={"marks": prev_q_marks, "review_status": prev_review_status},
            new_value={"marks": teacher_marks, "review_status": "TEACHER_OVERRIDE"},
            source=reviewer,
            reason=reason,
        )
    )
    
    # Recalculate Assessment Totals & Percentage
    recalculate_assessment_totals(struct_res)
    struct_res.updated_at = now_iso
    
    # If assessment was already FINALIZED, create new snapshot & revision for post-finalization edit!
    if was_finalized:
        snap_payload = build_snapshot_payload(struct_res, reviewer, f"Post-finalization override: {reason}")
        snap_path, snap_hash = store.save_snapshot(struct_res.assessment_id, struct_res.revision_index, snap_payload)
        
        revision = AssessmentRevision(
            revision_id=f"ver_{struct_res.revision_index}_{datetime.now(timezone.utc).strftime('%H%M%S')}",
            revision_index=struct_res.revision_index,
            timestamp=now_iso,
            final_awarded_marks=struct_res.final_awarded_marks,
            percentage=struct_res.percentage,
            finalized_by=reviewer,
            reason=f"Post-finalization override: {reason}",
            snapshot_hash=snap_hash,
            snapshot_file=snap_path,
        )
        struct_res.version_history.append(revision)
    
    # Save back to store & AssessmentResult wrapper
    assessment_res.structured_result = struct_res
    store.save_result(assessment_res)
    
    return struct_res


def build_snapshot_payload(struct_res: StructuredAssessmentResult, finalized_by: str, reason: str) -> dict:
    """Builds a complete immutable JSON snapshot dict for a finalized assessment revision."""
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "assessment_id": struct_res.assessment_id,
        "revision_index": struct_res.revision_index,
        "assessment_status": "FINALIZED",
        "finalization_timestamp": now_iso,
        "finalized_by": finalized_by,
        "reason": reason,
        "total_questions": struct_res.total_questions,
        "answered_questions": struct_res.answered_questions,
        "unanswered_questions": struct_res.unanswered_questions,
        "unmatched_answers_count": struct_res.unmatched_answers_count,
        "total_max_marks": struct_res.total_max_marks,
        "ai_awarded_marks": struct_res.ai_awarded_marks,
        "teacher_adjusted_marks": struct_res.teacher_adjusted_marks,
        "final_awarded_marks": struct_res.final_awarded_marks,
        "percentage": struct_res.percentage,
        "overall_confidence": struct_res.overall_confidence,
        "grading_statistics": struct_res.grading_statistics,
        "question_results": [q.model_dump() for q in struct_res.question_results],
        "audit_trail": [a.model_dump() for a in struct_res.audit_trail],
    }


def finalize_assessment(
    assessment_res: AssessmentResult,
    reviewer: str = "Teacher",
    reason: str = "Teacher finalized assessment results",
) -> StructuredAssessmentResult:
    """
    Finalizes assessment status (`FINALIZED`), freezes scores, builds immutable JSON snapshot,
    computes SHA-256 content hash, and records revision version history.
    """
    struct_res = assessment_res.structured_result
    if not struct_res:
        raise ValueError("Assessment structured result not found")
        
    struct_res.assessment_id = assessment_res.assessment_id
    now_iso = datetime.now(timezone.utc).isoformat()
    prev_status = struct_res.assessment_status
    
    # Recalculate totals
    recalculate_assessment_totals(struct_res)
    struct_res.assessment_status = "FINALIZED"
    struct_res.updated_at = now_iso
    
    # Build & Save Immutable Snapshot
    snap_payload = build_snapshot_payload(struct_res, reviewer, reason)
    snap_path, snap_hash = store.save_snapshot(struct_res.assessment_id, struct_res.revision_index, snap_payload)
    
    # Create Revision Record with snapshot metadata
    revision = AssessmentRevision(
        revision_id=f"ver_{struct_res.revision_index}_{datetime.now(timezone.utc).strftime('%H%M%S')}",
        revision_index=struct_res.revision_index,
        timestamp=now_iso,
        final_awarded_marks=struct_res.final_awarded_marks,
        percentage=struct_res.percentage,
        finalized_by=reviewer,
        reason=reason,
        snapshot_hash=snap_hash,
        snapshot_file=snap_path,
    )
    struct_res.version_history.append(revision)
    
    # Audit Finalization Event
    struct_res.audit_trail.append(
        create_audit_event(
            assessment_id=struct_res.assessment_id,
            event_type="FINAL_RESULT_UPDATED",
            previous_value={"status": prev_status},
            new_value={"status": "FINALIZED", "final_marks": struct_res.final_awarded_marks, "percentage": struct_res.percentage, "snapshot_hash": snap_hash},
            source=reviewer,
            reason=reason,
        )
    )
    
    assessment_res.structured_result = struct_res
    store.save_result(assessment_res)
    return struct_res


def recalculate_assessment_totals(struct_res: StructuredAssessmentResult) -> None:
    """Recalculates total max marks, teacher adjusted total, final marks, percentage, and review queue."""
    total_max = sum(q.max_marks for q in struct_res.question_results)
    ai_awarded = sum(q.original_ai_marks for q in struct_res.question_results)
    final_awarded = sum(q.awarded_marks for q in struct_res.question_results)
    
    teacher_adj = sum(q.teacher_adjusted_marks for q in struct_res.question_results if q.teacher_adjusted_marks is not None)
    has_any_override = any(q.teacher_adjusted_marks is not None for q in struct_res.question_results)
    
    pct = round((final_awarded / total_max) * 100, 2) if total_max > 0 else 0.0
    review_queue = build_review_queue(struct_res.question_results)
    
    struct_res.total_max_marks = total_max
    struct_res.ai_awarded_marks = ai_awarded
    struct_res.teacher_adjusted_marks = teacher_adj if has_any_override else None
    struct_res.final_awarded_marks = final_awarded
    struct_res.percentage = pct
    struct_res.questions_needing_review = review_queue["pending_count"]
    struct_res.review_summary = review_queue
