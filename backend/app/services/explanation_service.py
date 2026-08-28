"""
Explainable Evaluation Service.
Provides human-readable, evidence-grounded explainability tracing:
Marks -> Criterion -> Evidence -> Answer Region -> Page + Bounding Box.
Reuses exact geometry produced by Step 2.
"""
from __future__ import annotations
from typing import Dict, Any, List
from app.models.schemas import QuestionResult, CriterionResult, CriterionEvidence


def build_question_explanation(q_res: QuestionResult) -> Dict[str, Any]:
    """
    Consumes real Step 4 results to construct an explainable evaluation summary
    linking awarded marks to bounding boxes and criterion evidence.
    """
    g_details = q_res.grading.result_details if (q_res.grading and q_res.grading.result_details) else None
    mapped_ans = q_res.answer
    
    max_m = g_details.max_marks if g_details else (q_res.grading.max_score if q_res.grading else 2.0)
    awarded = g_details.awarded_marks if g_details else (q_res.grading.score if q_res.grading else 0.0)
    confidence = g_details.confidence if g_details else 0.50
    eval_method = g_details.evaluation_method if g_details else "local"
    routing = g_details.routing_decision if g_details else "LOCAL_CLEAR"
    
    # Extract source regions geometry from Step 2
    source_regions = []
    if mapped_ans and mapped_ans.regions:
        for reg in mapped_ans.regions:
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
            
    # Build criterion-level explainability breakdown
    criterion_results: List[CriterionResult] = []
    strengths: List[str] = []
    partial_points: List[str] = []
    missing_points: List[str] = []
    
    raw_criteria = g_details.criteria if g_details else []
    for ev in raw_criteria:
        c_res = CriterionResult(
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
        criterion_results.append(c_res)
        
        if ev.status == "present":
            strengths.append(f"Satisfied criterion '{ev.description}' ({ev.awarded_marks}/{ev.max_marks} marks)")
        elif ev.status == "partially_present":
            partial_points.append(f"Partial credit for '{ev.description}' ({ev.awarded_marks}/{ev.max_marks} marks)")
        elif ev.status == "missing":
            missing_points.append(f"Missing required concept '{ev.description}' (0/{ev.max_marks} marks)")
        elif ev.status == "contradicted":
            missing_points.append(f"Contradiction identified in '{ev.description}' ({ev.notes or '0 marks'})")

    return {
        "question_id": q_res.id,
        "question_number": q_res.number,
        "question_text": q_res.text,
        "awarded_marks": awarded,
        "max_marks": max_m,
        "confidence": confidence,
        "evaluation_method": eval_method,
        "routing_decision": routing,
        "strengths": strengths,
        "partial_points": partial_points,
        "missing_points": missing_points,
        "criterion_results": criterion_results,
        "source_regions": source_regions,
        "explanation_summary": (
            f"VedaAI evaluated Question Q{q_res.number} via {eval_method} engine. "
            f"Awarded {awarded}/{max_m} marks with {int(confidence * 100)}% evaluation confidence."
        ),
    }
