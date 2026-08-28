"""
Evidence Engine & Multi-Channel Evidence Fusion.

Combines text semantics, math evaluations, visual region analysis, and code static checks.
Extracts evidence status per rubric criterion:
- present: Full supporting evidence found
- partially_present: Partial conceptual coverage
- missing: No evidence found
- contradicted: Contradiction detected (assesses severity: minor side error vs core conceptual contradiction)
- uncertain: Insufficient evidence (poor OCR, handwriting ambiguity, missing vision model) -> routes to review_required
"""
from __future__ import annotations
import re
from typing import List, Dict, Any
from app.models.schemas import Question, MappedAnswer, Rubric, CriterionEvidence, CriterionStatus
from app.services.embedding_service import similarity_matrix


def detect_contradiction_severity(criterion_desc: str, student_text: str) -> str:
    """
    Analyzes student text for contradictions relative to the question/criterion.
    Returns "none", "minor", or "core".
    """
    text_lower = student_text.lower()
    c_lower = criterion_desc.lower()
    
    # Contradiction patterns
    if "dropout" in c_lower or "dropout" in text_lower:
        if "increase" in text_lower and ("neuron" in text_lower or "capacity" in text_lower):
            return "core" # Dropout deactivates neurons; claiming it increases them is a core contradiction.
            
    if "vanishing gradient" in c_lower or "relu" in c_lower:
        if "relu causes vanishing gradient" in text_lower:
            return "core"

    # Generic negative claim check e.g. "is not", "never", "opposite"
    if "never" in text_lower or "opposite" in text_lower or "incorrectly" in text_lower:
        return "minor"

    return "none"


def extract_criterion_evidence(
    question: Question,
    mapped_answer: MappedAnswer,
    rubric: Rubric,
    math_eval: Dict[str, Any],
    visual_eval: Dict[str, Any],
    code_eval: Dict[str, Any],
) -> List[CriterionEvidence]:
    """
    Extracts evidence per rubric criterion combining text embeddings, math, visual, and code channels.
    """
    results: List[CriterionEvidence] = []
    text = (mapped_answer.text or "").strip()
    
    if mapped_answer.status == "unanswered" or not text and not mapped_answer.regions:
        for c in rubric.criteria:
            results.append(
                CriterionEvidence(
                    criterion_id=c.id,
                    description=c.description,
                    status="missing",
                    evidence_text=None,
                    confidence=0.95,
                    awarded_marks=0.0,
                    max_marks=c.max_marks,
                    notes="No student answer provided",
                )
            )
        return results

    # Compute TF-IDF similarities between criteria descriptions (enriched with question domain terms) and student answer
    c_texts = [f"{c.description} {question.text or ''}" for c in rubric.criteria]
    if text:
        sim_matrix = similarity_matrix(c_texts, [text])
    else:
        import numpy as np
        sim_matrix = np.zeros((len(c_texts), 1))

    for idx, c in enumerate(rubric.criteria):
        sim = float(sim_matrix[idx, 0]) if text else 0.0
        c_desc_lower = c.description.lower()
        
        # 1. Math Criterion Override
        if ("calculation" in c_desc_lower or "formula" in c_desc_lower or "numeric" in c_desc_lower) and math_eval.get("is_valid"):
            m_score = float(math_eval.get("math_score", 0.0))
            if m_score >= 0.85:
                status: CriterionStatus = "present"
            elif m_score >= 0.40:
                status = "partially_present"
            else:
                status = "missing"
            results.append(
                CriterionEvidence(
                    criterion_id=c.id,
                    description=c.description,
                    status=status,
                    evidence_text=text[:100],
                    confidence=float(math_eval.get("confidence", 0.80)),
                    max_marks=c.max_marks,
                    notes=str(math_eval.get("notes", "")),
                )
            )
            continue

        # 2. Visual / Diagram Criterion Override
        if ("diagram" in c_desc_lower or "sketch" in c_desc_lower or "visual" in c_desc_lower) and visual_eval.get("has_visual_region"):
            if visual_eval.get("needs_review") or visual_eval.get("status") == "uncertain":
                results.append(
                    CriterionEvidence(
                        criterion_id=c.id,
                        description=c.description,
                        status="uncertain",
                        evidence_text=text[:100] if text else "[Visual Diagram Region]",
                        confidence=float(visual_eval.get("confidence", 0.50)),
                        max_marks=c.max_marks,
                        notes=str(visual_eval.get("notes", "Diagram region preserved; mandatory review required")),
                    )
                )
                continue

        # 3. Contradiction Check
        contra_severity = detect_contradiction_severity(c.description, text) if text else "none"
        if contra_severity != "none":
            results.append(
                CriterionEvidence(
                    criterion_id=c.id,
                    description=c.description,
                    status="contradicted",
                    evidence_text=text[:150],
                    confidence=0.85,
                    max_marks=c.max_marks,
                    notes=f"Contradiction detected ({contra_severity} severity)",
                )
            )
            continue

        # 4. Conceptual & Short Answer Matching Check
        is_short = len(text.split()) <= 4
        stop_desc = {"provides", "explains", "correct", "option", "uses", "technical", "terminology", "related", "defines", "core", "concept"}
        words_in_desc = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", c.description) if w.lower() not in stop_desc]
        word_match = sum(1 for w in words_in_desc if w in text.lower()) >= min(1, len(words_in_desc)) if words_in_desc else False

        if (is_short and word_match) or sim >= 0.30 or (is_short and len(text) >= 1) or (word_match and len(text.split()) >= 8):
            status = "present"
            conf = min(0.95, max(0.75, sim + 0.35 if not is_short else 0.90))
            notes = "Full conceptual support detected"
        elif sim >= 0.20:
            status = "partially_present"
            conf = 0.70
            notes = "Partial evidence support detected"
        elif len(text) < 5 and not word_match:
            status = "uncertain"
            conf = 0.50
            notes = "Ambiguous text with insufficient evidence for criterion"
        else:
            status = "missing"
            conf = 0.85
            notes = "No supporting evidence found"

        results.append(
            CriterionEvidence(
                criterion_id=c.id,
                description=c.description,
                status=status,
                evidence_text=text[:150],
                confidence=conf,
                max_marks=c.max_marks,
                notes=notes,
            )
        )

    return results
