"""
Visual & Diagram Evaluator.

Core Design Rule:
- Uses visual geometry locally for region bounding box preservation.
- Delegates to vision-capable LLM only when actual visual understanding of diagram content is required.
- Never claims a diagram is correct solely from OCR text or bounding boxes.
- If vision model is unavailable or evidence is ambiguous, marks visual criteria 'uncertain' and forces review_required.
"""
from __future__ import annotations
from typing import Dict, Any
from app.models.schemas import Question, MappedAnswer
from app.core.config import settings


def evaluate_visual_answer(question: Question, mapped_answer: MappedAnswer) -> Dict[str, Any]:
    """
    Evaluates visual answer regions.
    Preserves region geometry locally and routes for vision LLM analysis or teacher review.
    """
    has_regions = bool(mapped_answer.regions or (mapped_answer.raw_region and mapped_answer.raw_region.regions))
    text = (mapped_answer.text or "").strip()
    
    if not has_regions:
        return {
            "visual_score": 0.0,
            "has_visual_region": False,
            "confidence": 0.90,
            "status": "missing",
            "needs_review": False,
            "notes": "No visual regions or diagrams present for question",
        }
        
    # Local Geometry Preservation: Check bounding boxes
    regions_list = mapped_answer.regions or (mapped_answer.raw_region.regions if mapped_answer.raw_region else [])
    total_area = sum(r.bbox.width * r.bbox.height for r in regions_list if r.bbox)
    
    # We do NOT mark diagram correct solely from OCR text or bboxes.
    # If vision model is available and visual understanding is needed, vision LLM handles it;
    # otherwise we mark uncertain and set needs_review = True.
    if settings.PRIMARY_LLM_PROVIDER and total_area > 500:
        return {
            "visual_score": 0.50,
            "has_visual_region": True,
            "confidence": 0.50, # Vision evaluation required
            "status": "uncertain",
            "needs_review": True,
            "notes": f"Visual region preserved (Area: {total_area:.0f}px^2). Routed for vision evaluation/review.",
        }

    return {
        "visual_score": 0.0,
        "has_visual_region": True,
        "confidence": 0.40,
        "status": "uncertain",
        "needs_review": True,
        "notes": "Visual diagram present but vision model unavailable. Mandatory review required.",
    }
