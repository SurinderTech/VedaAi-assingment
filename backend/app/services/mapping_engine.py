"""
Step 3 — Intelligent Question ↔ Answer Mapping Engine.

Architecture & Features:
1. Candidate Generation & Normalization
2. Independent Multi-Signal Evidence Evaluation (Anchor, Semantic, Structural, Spatial, Order)
3. Conflict Detection (Mismatched Anchors, Semantic vs Spatial Mismatch)
4. Global Bipartite Assignment with Null-Nodes (scipy.optimize.linear_sum_assignment)
5. Competition & Ambiguity Margin Analysis (best_score vs second_best_score)
6. Targeted LLM Ambiguity Resolution (Invoked ONLY for conflicting/ambiguous candidate sets)
"""
from __future__ import annotations
import asyncio
import json
import re
from typing import List, Optional, Tuple, Dict, Set, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.config import settings
from app.models.schemas import (
    Question,
    AnswerCandidate,
    AnswerRegion,
    MappedAnswer,
    Region,
    UnmatchedAnswer,
)
from app.services.embedding_service import similarity_matrix
from app.services.semantic_retrieval_service import get_semantic_candidates
from app.services.llm_provider import llm_complete


def _normalize_anchor_key(num_str: Optional[str]) -> str:
    """Normalizes variations like 'Q1(a)', '1(a)', '1a', '1.a', 'Q7', '7.' -> '1(a)' / '7'."""
    if not num_str:
        return ""
    s = num_str.strip().lower()
    is_or = "or" in s
    s = re.sub(r"^(?:ans(?:wer)?\.?\s*)?(?:q(?:uestion)?\.?\s*)?", "", s)
    s = re.sub(r"\s*or\s*$", "", s).strip()

    # Subquestion format e.g. 1(a), 1.a, 1a, 1 (a)
    m_sub = re.match(r"^(\d{1,3})\s*[\.\s\-_]*\(?([a-z1-9])\)?$", s)
    if m_sub:
        norm = f"{m_sub.group(1)}({m_sub.group(2)})"
    else:
        # Main question format e.g. 7., 7
        m_main = re.match(r"^(\d{1,3})[\.\s]*$", s)
        if m_main:
            norm = m_main.group(1)
        else:
            norm = s

    return f"{norm}_or" if is_or else norm


def _coerce_to_answer_region(a: Union[AnswerCandidate, AnswerRegion]) -> AnswerRegion:
    """Converts legacy AnswerCandidate objects into AnswerRegion models if necessary."""
    if isinstance(a, AnswerRegion):
        return a
    return AnswerRegion(
        answer_id=a.answer_id,
        question_anchor=a.question_number,
        pages=list(set(r.page for r in a.regions)) if a.regions else [1],
        regions=a.regions,
        text=a.text,
        reading_order=getattr(a, "order_index", 0),
        confidence=0.9,
    )


def _evaluate_candidate_evidence(
    q: Question,
    q_idx: int,
    region: AnswerRegion,
    r_idx: int,
    semantic_sim: float,
    total_questions: int,
    total_regions: int,
    all_q_norms: Set[str],
    sim_matrix: np.ndarray = None,
) -> Dict[str, float | bool | str]:
    """
    Evaluates independent evidence signals between a Question and an AnswerRegion candidate.
    Signals: Anchor, Semantic, Structural, Spatial, Order.
    Detects conflicts when signals contradict.
    """
    q_norm = _normalize_anchor_key(q.number)
    a_anchor_raw = region.question_anchor
    a_norm = _normalize_anchor_key(a_anchor_raw) if a_anchor_raw else ""

    # 1. Anchor Evidence Signal
    anchor_score = 0.0
    is_foreign_anchor = False
    if a_norm:
        if a_norm == q_norm:
            anchor_score = 1.0
        elif a_norm.replace("_or", "") == q_norm.replace("_or", ""):
            anchor_score = 0.95
        else:
            anchor_score = 0.0
            if a_norm in all_q_norms:
                is_foreign_anchor = True  # Anchor points to ANOTHER valid question on test

    # 2. Semantic Similarity Signal
    semantic_score = float(np.clip(semantic_sim, 0.0, 1.0))

    # 3. Structural & Continuation Evidence Signal
    struct_score = 0.5
    if region.is_continuation:
        struct_score += 0.3
    if a_norm == q_norm:
        struct_score += 0.2
    struct_score = min(1.0, struct_score)

    # 4. Spatial Evidence Signal (Normalized Geometry)
    spatial_score = 0.5
    if region.regions:
        page_diff = abs(region.regions[0].page - q.page)
        spatial_score = max(0.0, 1.0 - 0.25 * page_diff)

    # 5. Order Alignment Evidence Signal
    order_score = 0.5
    if total_questions > 1 and total_regions > 1:
        norm_q_order = q_idx / max(1, total_questions - 1)
        norm_r_order = r_idx / max(1, total_regions - 1)
        order_diff = abs(norm_q_order - norm_r_order)
        order_score = max(0.0, 1.0 - 1.5 * order_diff)

    # 6. Conflict & Non-Existent Anchor Detection
    conflict_detected = False
    conflict_reason = ""

    if a_norm and a_norm != q_norm:
        if is_foreign_anchor:
            # Anchor specifies another question on test (e.g. Q8 vs Q7) -> Conflict
            conflict_detected = True
            conflict_reason = f"Explicit anchor '{a_anchor_raw}' specifies {a_norm} which differs from {q.number}"
        else:
            # Anchor specifies a question NOT on test (e.g. Q99) -> Reject as candidate for q
            return {
                "anchor_score": 0.0,
                "semantic_score": round(semantic_score, 3),
                "structural_score": round(struct_score, 3),
                "spatial_score": round(spatial_score, 3),
                "order_score": round(order_score, 3),
                "final_score": 0.0,
                "conflict_detected": False,
                "conflict_reason": "Non-existent question anchor",
            }

    if semantic_score > 0.85 and spatial_score < 0.25:
        conflict_detected = True
        conflict_reason = f"High semantic score ({semantic_score:.2f}) conflicts with spatial page distance"

    # Case C: Explicit anchor matches current question number (e.g. Q2) but text content strongly matches another question (e.g. Q1)
    if anchor_score >= 0.95 and semantic_score < 0.10 and len(region.text or "") > 15:
        if total_questions > 1:
            # Check if text strongly matches a different question on the exam
            other_sims = [float(sim_matrix[other_qi, r_idx]) for other_qi in range(total_questions) if other_qi != q_idx]
            if other_sims and max(other_sims) > 0.50:
                conflict_detected = True
                conflict_reason = f"Explicit anchor '{a_anchor_raw}' matches {q.number} but text content strongly matches another question ({max(other_sims):.2f})"

    # Multi-Signal Composite Score Calculation
    if not a_norm:
        # Unanchored student work (no explicit Q-number): redistribute anchor weight to semantic & structural signals
        w_anchor = 0.0
        w_semantic = 0.55
        w_struct = 0.25
        w_spatial = 0.15
        w_order = 0.05
    else:
        w_anchor = settings.MAPPING_ANCHOR_WEIGHT
        w_semantic = settings.MAPPING_SEMANTIC_WEIGHT
        w_struct = settings.MAPPING_STRUCTURAL_WEIGHT
        w_spatial = settings.MAPPING_SPATIAL_WEIGHT
        w_order = settings.MAPPING_ORDER_WEIGHT

    s_anch = round(anchor_score, 3)
    s_sem = round(semantic_score, 3)
    s_struct = round(struct_score, 3)
    s_spat = round(spatial_score, 3)
    s_ord = round(order_score, 3)

    raw_final_score = round(
        w_anchor * s_anch
        + w_semantic * s_sem
        + w_struct * s_struct
        + w_spatial * s_spat
        + w_order * s_ord,
        3,
    )

    conflict_penalty = 0.70 if conflict_detected else 1.00
    final_score = round(raw_final_score * conflict_penalty, 3)

    return {
        "anchor_score": s_anch,
        "semantic_score": s_sem,
        "structural_score": s_struct,
        "spatial_score": s_spat,
        "order_score": s_ord,
        "w_anchor": w_anchor,
        "w_semantic": w_semantic,
        "w_struct": w_struct,
        "w_spatial": w_spatial,
        "w_order": w_order,
        "raw_final_score": raw_final_score,
        "conflict_penalty": conflict_penalty,
        "final_score": final_score,
        "conflict_detected": conflict_detected,
        "conflict_reason": conflict_reason,
    }


async def map_answers(
    questions: List[Question],
    answers: List[Union[AnswerCandidate, AnswerRegion]],
    page_sizes: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Dict[str, MappedAnswer], List[UnmatchedAnswer]]:
    """
    Step 3 Main Entry Point: Intelligent Question ↔ Answer Mapping Engine.
    
    Returns:
    - result: Dict[question_id, MappedAnswer]
    - unmatched_answers: List[UnmatchedAnswer]
    """
    result: Dict[str, MappedAnswer] = {}
    if not questions:
        return result, []

    # Coerce input candidates to AnswerRegion instances
    regions: List[AnswerRegion] = [_coerce_to_answer_region(a) for a in answers]
    num_q = len(questions)
    num_r = len(regions)

    if num_r == 0:
        for q in questions:
            result[q.id] = MappedAnswer(
                status="unanswered",
                confidence=0.0,
                method="no_student_answer_regions",
                evidence_summary="No student answer regions found on answer booklet",
            )
        return result, []

    # 1. Compute Dense / TF-IDF Semantic Similarity Matrix (Step 8 Semantic Intelligence)
    sem_retrieval = get_semantic_candidates(questions, regions)
    sim_matrix = sem_retrieval.get("similarity_matrix")
    if sim_matrix is None or len(sim_matrix) == 0:
        q_texts = [q.text for q in questions]
        r_texts = [r.text for r in regions]
        sim_matrix = similarity_matrix(q_texts, r_texts)

    all_q_norms = {_normalize_anchor_key(q.number) for q in questions}

    # 2. Evaluate Candidate Evidence Matrix
    evidence_matrix: List[List[Dict[str, float | bool | str]]] = []
    score_matrix = np.zeros((num_q, num_r))

    for qi, q in enumerate(questions):
        row_ev = []
        for ri, r in enumerate(regions):
            ev = _evaluate_candidate_evidence(q, qi, r, ri, float(sim_matrix[qi, ri]), num_q, num_r, all_q_norms, sim_matrix)
            row_ev.append(ev)
            score_matrix[qi, ri] = float(ev["final_score"])
        evidence_matrix.append(row_ev)

    # 3. Global 1-to-1 Bipartite Matching with Null Nodes (scipy linear_sum_assignment)
    # Cost matrix padded with null nodes so questions/answers can remain unassigned
    null_cost = 1.0 - settings.MAPPING_REVIEW_THRESHOLD  # e.g., 1.0 - 0.45 = 0.55
    cost_matrix = np.full((num_q, num_r + num_q), null_cost)

    for qi in range(num_q):
        for ri in range(num_r):
            score = score_matrix[qi, ri]
            if score >= 0.10:
                cost_matrix[qi, ri] = 1.0 - score
            else:
                cost_matrix[qi, ri] = 1.0  # Incompatible candidate

        # Null assignment node for question qi
        cost_matrix[qi, num_r + qi] = null_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Map assignments
    assigned_region_indices: Set[int] = set()
    mapped_q_indices: Set[int] = set()

    q_assignments: Dict[int, int] = {}
    for qi, c_idx in zip(row_ind, col_ind):
        if c_idx < num_r:
            assigned_score = score_matrix[qi, c_idx]
            if assigned_score >= settings.MAPPING_REVIEW_THRESHOLD:
                q_assignments[qi] = c_idx
                assigned_region_indices.add(c_idx)
                mapped_q_indices.add(qi)

    # 4. Competition Analysis, Margin Check & Conflict Inspection
    for qi, q in enumerate(questions):
        if qi in q_assignments:
            ri = q_assignments[qi]
            r = regions[ri]
            ev = evidence_matrix[qi][ri]
            best_score = float(ev["final_score"])

            # Compute second best candidate score for question q
            other_scores = [score_matrix[qi, k] for k in range(num_r) if k != ri]
            second_best_score = max(other_scores) if other_scores else 0.0
            score_margin = round(best_score - second_best_score, 3)

            conflict_detected = bool(ev["conflict_detected"])
            is_ambiguous = (score_margin < settings.MAPPING_AMBIGUITY_DELTA and num_r > 1) or (best_score < settings.MAPPING_HIGH_CONFIDENCE_THRESHOLD)

            needs_review = conflict_detected or is_ambiguous
            status = "review_required" if needs_review else "matched"
            method = "multi_signal_global_bipartite"
            if ev["anchor_score"] >= 0.95 and not conflict_detected:
                method = "explicit_question_anchor"

            ev_summary = (
                f"Anchor: {ev['anchor_score']:.3f} (w={ev['w_anchor']:.2f}) | "
                f"Semantic: {ev['semantic_score']:.3f} (w={ev['w_semantic']:.2f}) | "
                f"Struct: {ev['structural_score']:.3f} (w={ev['w_struct']:.2f}) | "
                f"Spatial: {ev['spatial_score']:.3f} (w={ev['w_spatial']:.2f}) | "
                f"Order: {ev['order_score']:.3f} (w={ev['w_order']:.2f}) | "
                f"Raw: {ev['raw_final_score']:.3f} | Penalty: {ev['conflict_penalty']:.2f} | "
                f"Final: {ev['final_score']:.3f} | Margin: {score_margin:.3f}"
            )
            if conflict_detected:
                ev_summary += f" [CONFLICT: {ev['conflict_reason']}]"

            # 5. Targeted LLM Fallback for Ambiguous / Conflicting Cases
            if needs_review and settings.PRIMARY_LLM_PROVIDER:
                llm_resolved = await _resolve_ambiguity_with_llm(q, r, ev_summary)
                if llm_resolved is not None:
                    best_score = max(best_score, llm_resolved)
                    needs_review = False
                    status = "matched"
                    method = "multi_signal_plus_llm"

            result[q.id] = MappedAnswer(
                status=status,
                answer_id=r.answer_id,
                text=r.text,
                confidence=best_score,
                method=method,
                regions=r.regions,
                anchor_score=float(ev["anchor_score"]),
                semantic_score=float(ev["semantic_score"]),
                structural_score=float(ev["structural_score"]),
                spatial_score=float(ev["spatial_score"]),
                order_score=float(ev["order_score"]),
                w_anchor=float(ev["w_anchor"]),
                w_semantic=float(ev["w_semantic"]),
                w_structural=float(ev["w_struct"]),
                w_spatial=float(ev["w_spatial"]),
                w_order=float(ev["w_order"]),
                raw_final_score=float(ev["raw_final_score"]),
                conflict_penalty=float(ev["conflict_penalty"]),
                final_score=best_score,
                best_candidate_score=best_score,
                second_best_candidate_score=second_best_score,
                score_margin=score_margin,
                conflict_detected=conflict_detected,
                needs_review=needs_review,
                evidence_summary=ev_summary,
                raw_region=r,
            )
        else:
            # Question has no valid candidate above acceptance threshold -> UNANSWERED
            result[q.id] = MappedAnswer(
                status="unanswered",
                confidence=0.0,
                method="no_valid_candidate",
                evidence_summary="No student answer region met candidate acceptance threshold",
            )

    # 6. Unmatched Answer Regions (Leftover regions not assigned to any question)
    unmatched_answers: List[UnmatchedAnswer] = []
    for ri, r in enumerate(regions):
        if ri not in assigned_region_indices:
            unmatched_answers.append(
                UnmatchedAnswer(
                    answer_id=r.answer_id,
                    text=r.text,
                    regions=r.regions,
                    confidence=0.0,
                )
            )

    return result, unmatched_answers


async def _resolve_ambiguity_with_llm(
    q: Question, r: AnswerRegion, evidence_summary: str
) -> Optional[float]:
    """
    Targeted LLM Ambiguity Resolver: Invoked ONLY for genuine candidate ties or conflicts.
    Sends minimal targeted context and returns calibrated score or None on failure.
    """
    prompt = (
        "You are an expert exam evaluator resolving an ambiguous question-to-answer mapping.\n"
        f"Question Number: {q.number}\n"
        f"Question Text: {q.text}\n"
        f"Student Answer Text: {r.text}\n"
        f"Evidence Context: {evidence_summary}\n\n"
        "Determining if this student answer belongs to this question.\n"
        "Return ONLY a JSON object: {\"match\": true/false, \"confidence\": float_between_0_and_1}"
    )
    try:
        raw = await asyncio.wait_for(llm_complete(prompt), timeout=3.5)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if data.get("match") is True:
                return float(data.get("confidence", 0.85))
            else:
                return 0.20
        return None
    except Exception:
        # LLM fallback safety: returns None on timeout/error so local decision is preserved
        return None



