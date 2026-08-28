"""
Intelligent Answer Evaluation & Evidence-Based Grading Orchestrator.

Architecture:
Question -> Requirement Spec -> Content Type -> Local Evaluators (Math/Visual/Code/Embeddings)
-> Local Evidence Analysis -> Routing Decision (LOCAL_CLEAR vs LLM_REQUIRED/LLM_RECOMMENDED)
-> Minimal Context Payload -> LLM Evidence Evaluator -> Validation Layer (Ignores LLM 'marks')
-> Evidence Fusion -> Deterministic Marks Engine (Sole Final Authority) -> Grading Result & Token Accounting.
"""
from __future__ import annotations
import asyncio
from typing import Optional
from app.models.schemas import Question, MappedAnswer, Grading, GradingResult
from app.services.question_requirement_analyzer import analyze_question_requirements
from app.services.answer_type_detector import detect_answer_content_type
from app.services.rubric_engine import generate_rubric
from app.services.math_evaluator import evaluate_mathematical_answer
from app.services.visual_evaluator import evaluate_visual_answer
from app.services.code_evaluator import evaluate_code_answer
from app.services.evidence_engine import extract_criterion_evidence
from app.services.llm_evaluator import (
    determine_routing_decision,
    evaluate_evidence_with_llm,
    validate_llm_evidence,
    fuse_evidence,
)
from app.services.marks_engine import calculate_marks_and_confidence
from app.core.config import settings


async def generate_grading(
    question: Question,
    mapped_answer: MappedAnswer,
    document_llm_calls: int = 0,
) -> Grading:
    """
    Step 4 Entry Point: Performs intelligent evidence-based grading with explicit LLM routing,
    evidence validation, evidence fusion, and deterministic marks authority.
    """
    # 1. Question Requirement Analysis & Structural Marks Resolution
    req_spec = analyze_question_requirements(question)
    
    # 2. Answer Content Type Detection
    content_type = detect_answer_content_type(mapped_answer)
    
    # Handle Unanswered Questions directly
    if mapped_answer.status == "unanswered" or not (mapped_answer.text or "").strip() and not mapped_answer.regions:
        max_m = req_spec.max_marks or 2.0
        g_result = GradingResult(
            question_id=question.id,
            answer_id=mapped_answer.answer_id,
            max_marks=max_m,
            awarded_marks=0.0,
            confidence=0.95,
            status="unanswered",
            needs_review=False,
            answer_type=req_spec.expected_answer_type,
            content_type=content_type,
            evaluation_method="local",
            routing_decision="LOCAL_CLEAR_WITH_HIGH_CONFIDENCE",
            escalation_reason="unanswered",
            llm_used=False,
            total_questions=1,
            local_evaluations=1,
            llm_evaluations=0,
            llm_calls_avoided=1,
            estimated_tokens_saved=1200,
            token_provenance="estimated",
            missing_evidence=["No student response written on answer sheet."],
            feedback=f"No student answer submitted for Question Q{question.number}.",
        )
        return Grading(
            score=0.0,
            max_score=max_m,
            strengths=[],
            missing_points=["No response submitted on answer sheet."],
            feedback=g_result.feedback,
            result_details=g_result,
        )

    # 3. Rubric Generation
    rubric = await generate_rubric(question, req_spec)
    
    # 4. Local Deterministic Evaluators
    math_eval = evaluate_mathematical_answer(question, mapped_answer) if (req_spec.has_numerical_requirement or content_type in ("formula", "mathematical_work", "number")) else {}
    visual_eval = evaluate_visual_answer(question, mapped_answer) if (req_spec.has_diagram_requirement or content_type in ("diagram", "visual_only")) else {}
    code_eval = evaluate_code_answer(question, mapped_answer) if (req_spec.has_code_requirement or content_type == "code") else {}

    # 5. Local Evidence Extraction
    local_evidence = extract_criterion_evidence(question, mapped_answer, rubric, math_eval, visual_eval, code_eval)
    
    # 6. Explicit Routing Decision (checking document LLM budget)
    routing_decision, escalation_reason = determine_routing_decision(
        mapped_answer, req_spec, content_type, local_evidence, math_eval, visual_eval, code_eval, llm_call_count=document_llm_calls
    )

    # 7. LLM Escalation & Failure Handling
    llm_used = False
    llm_provider = None
    eval_method = "local"
    llm_eval_count = 0
    llm_calls_avoided = 1
    llm_fail_count = 0
    est_in_tokens = 0
    est_out_tokens = 0
    est_saved_tokens = 1200

    should_call_llm = (
        routing_decision in ("LLM_REQUIRED", "LLM_RECOMMENDED")
        and settings.GRADING_LLM_ENABLED
        and bool(settings.PRIMARY_LLM_PROVIDER)
    )

    if should_call_llm:
        llm_res = await evaluate_evidence_with_llm(
            question, mapped_answer, req_spec, rubric, local_evidence, escalation_reason
        )
        if llm_res.get("success"):
            llm_used = True
            llm_provider = llm_res.get("provider") or settings.PRIMARY_LLM_PROVIDER
            eval_method = "local+llm"
            llm_eval_count = 1
            llm_calls_avoided = 0
            est_in_tokens = 450
            est_out_tokens = 150
            est_saved_tokens = 0
            
            validated_llm_ev = validate_llm_evidence(llm_res["data"], rubric)
            final_criteria_evidence = fuse_evidence(local_evidence, validated_llm_ev)
        else:
            # Failure Fallback Path: preserve local evidence, route to review, mark local_fallback
            eval_method = "local_fallback"
            llm_fail_count = 1
            # Flag uncertain for ambiguous criteria so needs_review = True
            final_criteria_evidence = []
            for ev in local_evidence:
                if ev.status == "missing" and (mapped_answer.text or "").strip():
                    ev.status = "uncertain"
                    ev.notes = f"LLM evaluation failed ({llm_res.get('failure_reason')}); local fallback flagged for review"
                final_criteria_evidence.append(ev)
    else:
        final_criteria_evidence = local_evidence

    # 8. Deterministic Marks Engine (SOLE FINAL AUTHORITY FOR AWARDED MARKS)
    calc_res = calculate_marks_and_confidence(rubric, final_criteria_evidence)
    
    awarded = calc_res["awarded_marks"]
    max_m = calc_res["max_marks"]
    confidence = calc_res["confidence"]
    needs_review = calc_res["needs_review"]
    status = calc_res["status"]
    
    # If fallback occurred, enforce review_required
    if eval_method == "local_fallback":
        needs_review = True
        status = "review_required"
        confidence = min(confidence, 0.52)
        
    # Generate Feedback String
    feedback_parts = [f"Score: {awarded}/{max_m} (Confidence: {int(confidence * 100)}%)."]
    if calc_res["correct_evidence"]:
        feedback_parts.append("Key Strengths: " + "; ".join(calc_res["correct_evidence"][:2]))
    if calc_res["incorrect_evidence"]:
        feedback_parts.append("Notes: " + "; ".join(calc_res["incorrect_evidence"][:2]))
    if calc_res["missing_evidence"]:
        feedback_parts.append("Missing: " + "; ".join(calc_res["missing_evidence"][:2]))
    if calc_res["uncertain_evidence"]:
        feedback_parts.append("Uncertain: " + "; ".join(calc_res["uncertain_evidence"][:2]))
        
    feedback_str = " ".join(feedback_parts)

    g_result = GradingResult(
        question_id=question.id,
        answer_id=mapped_answer.answer_id,
        max_marks=max_m,
        awarded_marks=awarded,
        confidence=confidence,
        status=status,
        needs_review=needs_review,
        answer_type=req_spec.expected_answer_type,
        content_type=content_type,
        criteria=final_criteria_evidence,
        correct_evidence=calc_res["correct_evidence"],
        missing_evidence=calc_res["missing_evidence"],
        incorrect_evidence=calc_res["incorrect_evidence"],
        partial_evidence=calc_res["partial_evidence"],
        uncertain_evidence=calc_res["uncertain_evidence"],
        semantic_score=float(mapped_answer.semantic_score),
        mathematical_score=float(math_eval.get("math_score", 0.0)),
        visual_score=float(visual_eval.get("visual_score", 0.0)),
        code_score=float(code_eval.get("code_score", 0.0)),
        evaluation_method=eval_method,
        routing_decision=routing_decision,
        escalation_reason=escalation_reason,
        llm_used=llm_used,
        llm_provider=llm_provider,
        total_questions=1,
        local_evaluations=1 if not llm_used else 0,
        llm_evaluations=llm_eval_count,
        llm_calls_avoided=llm_calls_avoided,
        estimated_input_tokens=est_in_tokens,
        estimated_output_tokens=est_out_tokens,
        estimated_tokens_saved=est_saved_tokens,
        token_provenance="estimated",
        llm_failure_count=llm_fail_count,
        feedback=feedback_str,
    )

    return Grading(
        score=awarded,
        max_score=max_m,
        strengths=calc_res["correct_evidence"],
        missing_points=calc_res["missing_evidence"],
        feedback=feedback_str,
        result_details=g_result,
    )
