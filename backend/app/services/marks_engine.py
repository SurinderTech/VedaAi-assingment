"""
Marks & Partial Credit Engine — SOLE FINAL AUTHORITY FOR AWARDED MARKS.

Core Architecture Principles:
1. Converts validated criterion evidence into numerical awarded marks deterministically.
2. Nuanced Contradiction Impact: A minor misstatement reduces partial credit (e.g., 75% or 50%),
   while a core conceptual contradiction zeroes out that specific criterion.
3. Hard Rule for Insufficient Evidence: 'uncertain' status criteria trigger needs_review = True and
   lower overall evaluation confidence; uncertainty is NEVER converted into silent zero or full marks.
4. Separates Awarded Marks from Evaluation Confidence.
"""
from __future__ import annotations
from typing import List, Dict, Any
from app.models.schemas import Rubric, CriterionEvidence, AnswerStatus
from app.core.config import settings


def calculate_marks_and_confidence(rubric: Rubric, criteria_evidence: List[CriterionEvidence]) -> Dict[str, Any]:
    """
    Deterministically computes final awarded marks, evaluation confidence, and review flags.
    """
    total_max_marks = rubric.total_max_marks or 2.0
    total_awarded = 0.0
    
    confidences: List[float] = []
    has_uncertain = False
    has_contradiction = False
    has_core_contradiction = False
    
    correct_ev: List[str] = []
    missing_ev: List[str] = []
    incorrect_ev: List[str] = []
    partial_ev: List[str] = []
    uncertain_ev: List[str] = []
    
    for ev in criteria_evidence:
        c_max = ev.max_marks
        c_awarded = 0.0
        confidences.append(ev.confidence)
        
        if ev.status == "present":
            c_awarded = c_max
            correct_ev.append(f"✓ {ev.description}")
            
        elif ev.status == "partially_present":
            c_awarded = round(c_max * 0.60, 2)
            partial_ev.append(f"⚠ {ev.description} (Partial coverage)")
            
        elif ev.status == "contradicted":
            has_contradiction = True
            # Nuanced Contradiction Handling: Severity assessment
            notes_lower = (ev.notes or "").lower()
            if "core" in notes_lower:
                has_core_contradiction = True
                c_awarded = 0.0
                incorrect_ev.append(f"✗ Core contradiction: {ev.description}")
            else:
                # Minor side error: reduce partial credit to 50%
                c_awarded = round(c_max * 0.50, 2)
                incorrect_ev.append(f"⚠ Minor misstatement: {ev.description}")
                
        elif ev.status == "uncertain":
            has_uncertain = True
            # Hard Rule: Uncertainty does NOT invent 0 or full marks.
            c_awarded = round(c_max * 0.50, 2)
            uncertain_ev.append(f"? Insufficient evidence: {ev.description}")
            
        else:  # missing
            c_awarded = 0.0
            missing_ev.append(f"✗ Missing: {ev.description}")
            
        ev.awarded_marks = c_awarded
        total_awarded += c_awarded

    if has_core_contradiction:
        total_awarded = 0.0

    # Bound awarded marks within [0.0, total_max_marks]
    total_awarded = max(0.0, min(total_max_marks, round(total_awarded, 2)))
    
    # Overall evaluation confidence
    avg_confidence = sum(confidences) / max(1, len(confidences))
    if has_uncertain:
        avg_confidence = min(avg_confidence, 0.52)  # Force low confidence for uncertainty
    if has_contradiction:
        avg_confidence = min(avg_confidence, 0.68)
        
    avg_confidence = round(avg_confidence, 2)
    
    # Review Decision Logic
    needs_review = (
        has_uncertain
        or has_contradiction
        or avg_confidence < settings.GRADING_HIGH_CONFIDENCE_THRESHOLD
    )
    
    status: AnswerStatus = "review_required" if needs_review else "graded"
    
    return {
        "awarded_marks": total_awarded,
        "max_marks": total_max_marks,
        "confidence": avg_confidence,
        "needs_review": needs_review,
        "status": status,
        "correct_evidence": correct_ev,
        "missing_evidence": missing_ev,
        "incorrect_evidence": incorrect_ev,
        "partial_evidence": partial_ev,
        "uncertain_evidence": uncertain_ev,
    }
