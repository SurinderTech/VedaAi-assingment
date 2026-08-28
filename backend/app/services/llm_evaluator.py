"""
LLM Intelligent Evaluator, Routing Decision Engine, Evidence Validation, and Evidence Fusion Layer.

Core Principles:
1. Routing Decision Engine determines whether local evidence is sufficient or requires LLM escalation.
2. Constructs minimal targeted context payload (Question, max marks, rubric, student answer, escalation reason).
3. Evidence Validation Layer validates criteria IDs, status, confidence, and EXPLICITLY IGNORES any LLM-returned "marks" fields.
4. Evidence Fusion Layer merges local and LLM evidence.
5. Robust Failure Handling: timeouts, malformed JSON, or API errors fall back safely to local evidence without crashing.
"""
from __future__ import annotations
import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional
from app.models.schemas import Question, MappedAnswer, QuestionRequirementSpec, Rubric, CriterionEvidence, RoutingDecision, CriterionStatus
from app.services.llm_provider import llm_complete_json_with_provider
from app.core.config import settings


def determine_routing_decision(
    mapped_answer: MappedAnswer,
    req_spec: QuestionRequirementSpec,
    content_type: str,
    local_evidence: List[CriterionEvidence],
    math_eval: Dict[str, Any],
    visual_eval: Dict[str, Any],
    code_eval: Dict[str, Any],
    llm_call_count: int = 0,
) -> Tuple[RoutingDecision, str]:
    """
    Determines whether local evidence is sufficient (LOCAL_CLEAR) or requires escalation (LLM_REQUIRED / LLM_RECOMMENDED).
    Returns (RoutingDecision, escalation_reason).
    """
    text = (mapped_answer.text or "").strip()
    
    # 0. Check Document-Level LLM Budget Limit
    if llm_call_count >= settings.GRADING_LLM_MAX_CALLS_PER_DOCUMENT:
        return "REVIEW_REQUIRED", "document_llm_budget_exceeded"
    
    # 1. Unanswered or Empty Response -> LOCAL_CLEAR
    if mapped_answer.status == "unanswered" or not text and not mapped_answer.regions:
        return "LOCAL_CLEAR_WITH_HIGH_CONFIDENCE", "unanswered"
        
    # 2. MCQ or Short Standalone Number -> LOCAL_CLEAR
    if content_type in ("mcq_selection", "number") or req_spec.expected_answer_type in ("mcq", "one_word"):
        return "LOCAL_CLEAR_WITH_HIGH_CONFIDENCE", "clear_short_factual_or_mcq"
        
    # 3. Diagram requirement requiring visual understanding -> LLM_REQUIRED
    if req_spec.has_diagram_requirement or content_type in ("diagram", "visual_only"):
        if visual_eval.get("has_visual_region"):
            return "LLM_REQUIRED", "diagram_visual_understanding"

    # 4. Local Contradiction with uncertain severity -> LLM_REQUIRED
    has_contradiction = any(ev.status == "contradicted" for ev in local_evidence)
    if has_contradiction:
        return "LLM_REQUIRED", "contradiction_severity_uncertain"

    # 5. Local Evidence Ambiguity / Partial Coverage -> LLM_RECOMMENDED
    has_partial_or_uncertain = any(ev.status in ("partially_present", "uncertain") for ev in local_evidence)
    if has_partial_or_uncertain:
        return "LLM_RECOMMENDED", "semantic_ambiguity"

    # 6. Complex / Paraphrased Conceptual Answer -> LLM_RECOMMENDED
    if req_spec.expected_answer_type in ("explanation", "long_conceptual", "short_conceptual", "process", "comparison", "definition") and content_type not in ("mcq_selection", "number"):
        # If not a short 1-word match, escalate paraphrased reasoning for LLM semantic evaluation
        if len(text.split()) > 4:
            return "LLM_RECOMMENDED", "complex_conceptual_paraphrasing"

    # 7. Check overall local confidence
    avg_conf = sum(ev.confidence for ev in local_evidence) / max(1, len(local_evidence))
    if avg_conf < settings.GRADING_LLM_CONFIDENCE_THRESHOLD:
        return "LLM_REQUIRED", "local_confidence_low"

    return "LOCAL_CLEAR_WITH_HIGH_CONFIDENCE", "local_evidence_high_confidence"


async def evaluate_evidence_with_llm(
    question: Question,
    mapped_answer: MappedAnswer,
    req_spec: QuestionRequirementSpec,
    rubric: Rubric,
    local_evidence: List[CriterionEvidence],
    escalation_reason: str,
) -> Dict[str, Any]:
    """
    Sends minimal targeted context to LLM and requests structured criteria evidence JSON.
    Handles timeouts and failures gracefully without crashing.
    """
    total_marks = req_spec.max_marks or 2.0
    
    # Minimal targeted context payload
    payload = {
        "question": question.text,
        "max_marks": total_marks,
        "requirements": {
            "answer_type": req_spec.expected_answer_type,
            "has_diagram": req_spec.has_diagram_requirement,
            "has_math": req_spec.has_numerical_requirement,
        },
        "rubric": [{"id": c.id, "description": c.description, "max_marks": c.max_marks} for c in rubric.criteria],
        "student_answer": mapped_answer.text,
        "escalation_reason": escalation_reason,
    }
    
    prompt = (
        "You are an expert exam evaluator resolving evidence for a student response.\n"
        f"Context Payload:\n{json.dumps(payload, indent=2)}\n\n"
        "Evaluate each criterion in the rubric. Determine if student evidence is:\n"
        "- 'present': full supporting evidence found\n"
        "- 'partially_present': partial evidence or incomplete explanation\n"
        "- 'missing': no supporting evidence\n"
        "- 'contradicted': student statement directly contradicts the criterion\n"
        "- 'uncertain': unreadable, ambiguous, or insufficient evidence\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "criteria": [\n'
        '    {"criterion_id": "c1", "status": "present", "confidence": 0.90, "evidence": "..."},\n'
        '    {"criterion_id": "c2", "status": "partially_present", "confidence": 0.75, "evidence": "..."}\n'
        '  ],\n'
        '  "contradictions": [\n'
        '    {"criterion_id": "c2", "severity": "minor", "evidence": "..."}\n'
        '  ],\n'
        '  "overall_confidence": 0.85\n'
        "}"
    )
    
    try:
        data, provider_name = await asyncio.wait_for(
            llm_complete_json_with_provider(prompt),
            timeout=settings.GRADING_LLM_TIMEOUT_SECONDS
        )
        if isinstance(data, dict):
            return {"success": True, "data": data, "provider": provider_name}
        return {"success": False, "failure_reason": "Invalid JSON format returned", "provider": provider_name}
    except asyncio.TimeoutError:
        return {"success": False, "failure_reason": f"LLM timeout exceeding {settings.GRADING_LLM_TIMEOUT_SECONDS}s"}
    except Exception as e:
        return {"success": False, "failure_reason": f"LLM provider error: {str(e)}"}


def validate_llm_evidence(llm_response_data: Dict[str, Any], rubric: Rubric) -> List[CriterionEvidence]:
    """
    Validates LLM criteria IDs, status literals, confidence bounds [0, 1], and
    CRITICAL RULE: Explicitly IGNORES any LLM-returned 'marks' or 'score' fields!
    """
    valid_ids = {c.id: c for c in rubric.criteria}
    validated_list: List[CriterionEvidence] = []
    
    raw_criteria = llm_response_data.get("criteria", [])
    if not isinstance(raw_criteria, list):
        return []
        
    for raw_c in raw_criteria:
        if not isinstance(raw_c, dict):
            continue
            
        c_id = str(raw_c.get("criterion_id") or raw_c.get("id") or "")
        if c_id not in valid_ids:
            continue  # Reject hallucinated criteria IDs!

        rubric_c = valid_ids[c_id]
        
        status_str = str(raw_c.get("status", "uncertain")).lower()
        if status_str not in ("present", "partially_present", "missing", "contradicted", "uncertain"):
            status_str = "uncertain"
            
        conf = float(raw_c.get("confidence", 0.70))
        conf = max(0.0, min(1.0, conf)) # Bound to [0.0, 1.0]
        
        ev_text = str(raw_c.get("evidence") or raw_c.get("evidence_text") or "")[:150]
        
        notes = f"LLM evidence interpretation (status={status_str})"
        if status_str == "contradicted":
            raw_contra = llm_response_data.get("contradictions", [])
            is_core = any(
                isinstance(contra, dict) and contra.get("criterion_id") == c_id and contra.get("severity") == "core"
                for contra in raw_contra
            ) or "core" in ev_text.lower()
            
            if is_core:
                notes = "Core contradiction identified by LLM evidence"
            else:
                notes = "Minor contradiction identified by LLM evidence"
        
        # Note: We NEVER accept raw_c.get("marks") or raw_c.get("score")!
        
        validated_list.append(
            CriterionEvidence(
                criterion_id=c_id,
                description=rubric_c.description,
                status=status_str, # type: ignore
                evidence_text=ev_text if ev_text else None,
                confidence=conf,
                max_marks=rubric_c.max_marks,
                notes=notes,
                provenance="llm",
            )
        )
        
    return validated_list


def fuse_evidence(
    local_evidence: List[CriterionEvidence],
    llm_evidence: List[CriterionEvidence]
) -> List[CriterionEvidence]:
    """
    Fuses local evidence and validated LLM evidence with evidence provenance tracking:
    - Local & LLM agree -> provenance = "fused_agreement", boost confidence.
    - LLM resolves local uncertainty / paraphrased text -> provenance = "fused_resolution".
    - Disagreement / Conflict -> status = "uncertain", provenance = "conflict_flagged",
      dynamic confidence derived from severity rather than flat constant!
    """
    llm_map = {ev.criterion_id: ev for ev in llm_evidence}
    fused: List[CriterionEvidence] = []
    
    for loc_ev in local_evidence:
        llm_ev = llm_map.get(loc_ev.criterion_id)
        if not llm_ev:
            loc_ev.provenance = "local"
            fused.append(loc_ev)
            continue
            
        if loc_ev.status == llm_ev.status:
            # Agreement -> boost confidence & tag fused_agreement
            boosted_conf = round(min(0.98, max(loc_ev.confidence, llm_ev.confidence) + 0.10), 2)
            fused.append(
                CriterionEvidence(
                    criterion_id=loc_ev.criterion_id,
                    description=loc_ev.description,
                    status=loc_ev.status,
                    evidence_text=llm_ev.evidence_text or loc_ev.evidence_text,
                    confidence=boosted_conf,
                    max_marks=loc_ev.max_marks,
                    notes=f"Local and LLM evidence agreed ({loc_ev.status}); confidence boosted to {boosted_conf}",
                    provenance="fused_agreement",
                )
            )
        elif loc_ev.status in ("uncertain", "missing", "partially_present") and llm_ev.status in ("present", "partially_present") and llm_ev.confidence >= 0.80:
            # LLM high-confidence semantic resolution of local partial/missing coverage -> tag fused_resolution
            fused.append(
                CriterionEvidence(
                    criterion_id=loc_ev.criterion_id,
                    description=loc_ev.description,
                    status=llm_ev.status,
                    evidence_text=llm_ev.evidence_text or loc_ev.evidence_text,
                    confidence=llm_ev.confidence,
                    max_marks=loc_ev.max_marks,
                    notes=f"LLM semantic evaluation resolved criterion to {llm_ev.status} (conf={llm_ev.confidence})",
                    provenance="fused_resolution",
                )
            )
        elif llm_ev.status == "contradicted":
            # LLM identified contradiction -> tag conflict_flagged
            fused.append(
                CriterionEvidence(
                    criterion_id=loc_ev.criterion_id,
                    description=loc_ev.description,
                    status="contradicted",
                    evidence_text=llm_ev.evidence_text,
                    confidence=llm_ev.confidence,
                    max_marks=loc_ev.max_marks,
                    notes="LLM identified contradiction in student reasoning",
                    provenance="conflict_flagged",
                )
            )
        else:
            # Evidence Disagreement -> Derive dynamic confidence based on disagreement severity!
            if (loc_ev.status == "partially_present" and llm_ev.status == "present") or (loc_ev.status == "present" and llm_ev.status == "partially_present"):
                # Minor disagreement: evaluable with lower confidence
                dyn_conf = round(max(0.55, min(loc_ev.confidence, llm_ev.confidence) - 0.15), 2)
                notes_msg = f"Minor evidence disagreement (local: {loc_ev.status}, LLM: {llm_ev.status}); dynamic conf={dyn_conf}"
                final_status = "partially_present"
            else:
                # Major contradiction / direct conflict -> lower confidence & mandatory review!
                dyn_conf = round(min(0.40, loc_ev.confidence, llm_ev.confidence), 2)
                notes_msg = f"Major evidence conflict between local ({loc_ev.status}) and LLM ({llm_ev.status}); flagged for review"
                final_status = "uncertain"

            fused.append(
                CriterionEvidence(
                    criterion_id=loc_ev.criterion_id,
                    description=loc_ev.description,
                    status=final_status,
                    evidence_text=llm_ev.evidence_text or loc_ev.evidence_text,
                    confidence=dyn_conf,
                    max_marks=loc_ev.max_marks,
                    notes=notes_msg,
                    provenance="conflict_flagged",
                )
            )
            
    return fused
