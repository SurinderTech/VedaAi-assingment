"""
Step 11C — Intelligent Question & Structure Extraction Service.

Converts OCR blocks and Step 11A/11B document-understanding structures into a validated,
evidence-backed question set represented as a DocumentQuestionExtractionResult.

Core Principles:
1. Strict Zero-Hallucination Rule: Every character of question/option text originates 100% from original OCR source regions.
2. VLM Independence: Works reliably whether VLM is active, unavailable, or disabled.
3. Feature-Gated Integration & Safe Fallback: Controlled via INTELLIGENT_EXTRACTION_ENABLED.
4. Context-Aware UNCERTAIN Preservation: Ambiguous candidates are preserved in UNCERTAIN state & audit rather than deleted.
5. Authoritative MCQ Option Source Regions: Options store source_region_ids, source_regions, page, and bbox.
6. Multi-Column Geometry Reading Order: Detects reading order columns before ordering regions.
7. Document-Level Audit Architecture: Full diagnostic ExtractionAudit metadata object produced per extraction run.
"""
from __future__ import annotations
import re
import uuid
from typing import List, Dict, Optional, Tuple, Any, Set

from app.core.config import settings
from app.models.schemas import (
    Block,
    BBox,
    Region,
    Question,
    ExtractedOption,
    ExtractedSection,
    RejectionRecord,
    ExtractionAudit,
    DocumentQuestionExtractionResult,
    DocumentRegion,
    DocumentPage,
    DocumentUnderstandingResult,
    StructureHypothesis,
    VerificationState,
)
from app.services.document_understanding_service import DocumentUnderstandingService


# --- Option Syntax Matchers ---
OPTION_PREFIX_RE = re.compile(
    r"^\s*[\(\[]?\s*([A-Da-d1-4])\s*[\)\]\.\:]\s*(.*)$"
)

SUBQUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\:\-]?\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)$",
    re.IGNORECASE,
)

MAIN_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})\s*[\.\):\-]\s*(.*)$",
    re.IGNORECASE,
)

SECTION_HEADER_RE = re.compile(
    r"^\s*(?:SECTION|PART|GROUP)\s*[\-\–\:\s]*([A-Z0-9]{1,3})\s*$",
    re.IGNORECASE,
)

ADMIN_INSTRUCTION_KEYWORDS = [
    "time allowed", "maximum marks", "max marks", "general instructions",
    "instructions to candidates", "roll no", "total pages", "total questions",
    "attempt any", "all questions are compulsory", "duration", "paper code",
    "subject code", "b.tech", "m.tech", "b.sc", "m.sc", "reg. no"
]


def _compute_bounding_box(regions: List[DocumentRegion]) -> Optional[BBox]:
    """Computes unifying bounding box for a list of document regions."""
    if not regions:
        return None
    xs = [r.bbox.x for r in regions]
    ys = [r.bbox.y for r in regions]
    ws = [r.bbox.x + r.bbox.width for r in regions]
    hs = [r.bbox.y + r.bbox.height for r in regions]
    min_x, min_y = min(xs), min(ys)
    max_w, max_h = max(ws) - min_x, max(hs) - min_y
    return BBox(x=round(min_x, 1), y=round(min_y, 1), width=round(max_w, 1), height=round(max_h, 1))


def _detect_reading_columns(regions: List[DocumentRegion], page_width: float) -> Dict[str, int]:
    """
    Multi-Column Geometry Reading Order Detector.
    Detects whether page is multi-column (e.g. 2-column) based on region X-midpoints.
    Returns region_id -> column_index mapping.
    """
    column_map: Dict[str, int] = {}
    if not regions or page_width <= 0:
        for r in regions:
            column_map[r.region_id] = 0
        return column_map

    # Check if regions fall clearly into left vs right halves
    mid = page_width / 2.0
    left_count = sum(1 for r in regions if (r.bbox.x + r.bbox.width / 2.0) < mid and r.bbox.width < page_width * 0.6)
    right_count = sum(1 for r in regions if (r.bbox.x + r.bbox.width / 2.0) >= mid and r.bbox.width < page_width * 0.6)

    is_multi_column = (left_count >= 2 and right_count >= 2)

    for r in regions:
        if is_multi_column:
            x_center = r.bbox.x + r.bbox.width / 2.0
            col = 0 if x_center < mid else 1
            column_map[r.region_id] = col
        else:
            column_map[r.region_id] = 0

    return column_map


class IntelligentQuestionExtractionService:
    """
    Dedicated service for converting structured document understanding into a validated question set.
    """

    def __init__(self, doc_understanding_service: Optional[DocumentUnderstandingService] = None):
        self.doc_understanding_service = doc_understanding_service or DocumentUnderstandingService()

    def extract_validated_questions(
        self,
        blocks: List[Block],
        document_id: str = "doc_1",
        doc_understanding_result: Optional[DocumentUnderstandingResult] = None,
        page_sizes: Optional[Dict[int, List[float]]] = None,
    ) -> DocumentQuestionExtractionResult:
        """
        Executes 10-stage intelligent question & structure extraction.
        """
        if not blocks:
            return DocumentQuestionExtractionResult(
                document_id=document_id,
                questions=[],
                sections=[],
                uncertain_candidates=[],
                audit=ExtractionAudit(),
            )

        # Ensure DocumentUnderstandingResult exists and contains regions
        if doc_understanding_result is None or not doc_understanding_result.regions:
            vlm_stat = doc_understanding_result.vlm_status if doc_understanding_result else "NOT_CONFIGURED"
            doc_understanding_result = self.doc_understanding_service.process_document(
                blocks=blocks, document_id=document_id, page_sizes=page_sizes
            )
            if vlm_stat != "NOT_CONFIGURED":
                doc_understanding_result.vlm_status = vlm_stat

        regions = doc_understanding_result.regions
        audit = ExtractionAudit(candidate_count=len(regions))
        rejection_records: List[RejectionRecord] = []

        # ---------------------------------------------------------------------
        # Stage 1 & 2: Candidate Classification & Administrative Exclusion
        # ---------------------------------------------------------------------
        classified_sections: List[ExtractedSection] = []
        accepted_regions: List[DocumentRegion] = []
        uncertain_regions: List[DocumentRegion] = []
        option_regions: List[DocumentRegion] = []

        active_section: Optional[ExtractedSection] = None
        sections_by_id: Dict[str, ExtractedSection] = {}

        for reg in regions:
            text_low = reg.text.strip().lower()

            # 1. Section Header Check
            sec_m = SECTION_HEADER_RE.match(reg.text.strip())
            if sec_m or reg.region_type == "SECTION_HEADER":
                sec_title = f"Section-{sec_m.group(1).upper()}" if sec_m else reg.text.strip()
                sec_id = f"sec_{len(classified_sections)+1}_{sec_title.lower()}"
                active_section = ExtractedSection(
                    section_id=sec_id,
                    title=sec_title,
                    page=reg.page,
                    bbox=reg.bbox,
                    source_region_ids=[reg.region_id],
                )
                classified_sections.append(active_section)
                sections_by_id[sec_id] = active_section
                audit.section_count += 1
                rejection_records.append(
                    RejectionRecord(
                        region_id=reg.region_id,
                        ocr_text=reg.text[:60],
                        classification="SECTION_HEADER",
                        confidence=reg.confidence,
                        reason="Structural container for section, not a question.",
                    )
                )
                continue

            # 2. Administrative / Instruction / Metadata / Header / Footer Filtering
            is_header_kw = any(kw in text_low for kw in ["roll no", "b.tech", "m.tech", "semester", "subject code", "max. marks", "max marks", "time:", "page "])
            is_kv_header = bool(re.match(r"^[A-Za-z0-9\s\.\(\)\–\-]{2,35}\s*:\s*.+$", reg.text.strip()))
            is_true_header = (reg.region_type in ("HEADER", "FOOTER", "METADATA") and (is_header_kw or is_kv_header or reg.bbox.y < 15.0))

            is_admin_kw = any(kw in text_low for kw in ADMIN_INSTRUCTION_KEYWORDS)
            is_explicit_admin_header = ("general instructions" in text_low or "instructions to candidates" in text_low or "time allowed" in text_low or "maximum marks" in text_low or "roll no" in text_low)
            is_top_instruction = (reg.region_type == "INSTRUCTION" and reg.page == 1 and reg.bbox.y < 200 and is_admin_kw)

            if (
                is_true_header
                or is_explicit_admin_header
                or is_top_instruction
            ) and not MAIN_QUESTION_PREFIX_RE.match(reg.text.strip()) and not SUBQUESTION_PREFIX_RE.match(reg.text.strip()):
                audit.rejected_count += 1
                rejection_records.append(
                    RejectionRecord(
                        region_id=reg.region_id,
                        ocr_text=reg.text[:60],
                        classification=reg.region_type,
                        confidence=reg.confidence,
                        reason=f"Excluded non-question administrative/instruction content ({reg.region_type}).",
                    )
                )
                continue

            # 2b. Structural Visual Element Filtering (Table / Diagram / Figure Text Leakage Protection)
            is_visual_structure = (
                reg.region_type in ("TABLE", "DIAGRAM", "FIGURE")
                or bool(re.search(r"^\s*(?:Table|Figure|Fig\.)\s+\d+", reg.text.strip(), re.IGNORECASE))
                or bool(re.search(r"\|.+\|", reg.text.strip()))
            )
            if is_visual_structure and not MAIN_QUESTION_PREFIX_RE.match(reg.text.strip()) and not SUBQUESTION_PREFIX_RE.match(reg.text.strip()):
                audit.rejected_count += 1
                v_type = reg.region_type if reg.region_type in ("TABLE", "DIAGRAM", "FIGURE") else "TABLE/FIGURE"
                rejection_records.append(
                    RejectionRecord(
                        region_id=reg.region_id,
                        ocr_text=reg.text[:60],
                        classification=v_type,
                        confidence=reg.confidence,
                        reason=f"Excluded visual structural element ({v_type}) from question prose text.",
                    )
                )
                continue

            # 3. MCQ Option Check
            opt_m = OPTION_PREFIX_RE.match(reg.text.strip())
            if reg.region_type == "OPTION" or (opt_m and not MAIN_QUESTION_PREFIX_RE.match(reg.text.strip())):
                option_regions.append(reg)
                audit.option_count += 1
                continue

            # 4. Question / Subquestion / Uncertain Candidate Check
            if reg.verification_state == "UNCERTAIN" or reg.confidence < 0.65:
                uncertain_regions.append(reg)
                audit.uncertain_count += 1
                accepted_regions.append(reg)  # Preserved without deletion
            else:
                accepted_regions.append(reg)

        # ---------------------------------------------------------------------
        # Stage 8: Multi-Column Geometry Reading Order Reconstruction
        # ---------------------------------------------------------------------
        # Organize regions by page, then column, then y-coordinate
        regions_by_page: Dict[int, List[DocumentRegion]] = {}
        for r in accepted_regions:
            regions_by_page.setdefault(r.page, []).append(r)

        ordered_regions: List[DocumentRegion] = []
        for page_num in sorted(regions_by_page.keys()):
            p_regs = regions_by_page[page_num]
            p_width = 1000.0
            if doc_understanding_result and doc_understanding_result.pages:
                p_match = next((p for p in doc_understanding_result.pages if p.page_number == page_num), None)
                if p_match and p_match.width > 0:
                    p_width = p_match.width

            col_map = _detect_reading_columns(p_regs, p_width)
            # Sort regions by (column_index, y_min, x_min)
            sorted_p_regs = sorted(p_regs, key=lambda r: (col_map.get(r.region_id, 0), r.bbox.y, r.bbox.x))
            ordered_regions.extend(sorted_p_regs)

        # ---------------------------------------------------------------------
        # Stage 3, 4, 5, 6, 7, 9: Question Boundary Construction & MCQ Assembly
        # ---------------------------------------------------------------------
        extracted_questions: List[Question] = []
        uncertain_question_candidates: List[Question] = []
        curr_section_id: Optional[str] = None
        curr_section_title: Optional[str] = None

        curr_question: Optional[Question] = None
        order_counter = 0

        current_main_parent_id: Optional[str] = None

        for reg in ordered_regions:
            txt = reg.text.strip()
            if not txt:
                continue

            # Check for section association update
            for sec in classified_sections:
                if sec.page == reg.page and reg.bbox.y >= sec.bbox.y:
                    curr_section_id = sec.section_id
                    curr_section_title = sec.title

            sub_m = SUBQUESTION_PREFIX_RE.match(txt)
            main_m = MAIN_QUESTION_PREFIX_RE.match(txt)

            # --- Subquestion Candidate e.g. 11(a) or 1(b) ---
            if sub_m:
                main_num = sub_m.group(1)
                sub_let = sub_m.group(2).lower()
                q_text = sub_m.group(3).strip() or txt

                q_id = f"{document_id}:{reg.region_id}"
                disp_num = f"{main_num}({sub_let})"
                parent_id = current_main_parent_id or f"{document_id}:parent_{main_num}"

                q_obj = Question(
                    id=q_id,
                    number=disp_num,
                    text=txt,  # 100% original OCR text preserved
                    page=reg.page,
                    bbox=reg.bbox,
                    order_index=order_counter,
                    section=curr_section_title,
                    section_id=curr_section_id,
                    section_title=curr_section_title,
                    parent_question_id=parent_id,
                    question_type="SUBQUESTION",
                    source_region_ids=[reg.region_id],
                    source_regions=[Region(page=reg.page, bbox=reg.bbox)],
                    extraction_confidence=reg.confidence,
                    verification_state=reg.verification_state,
                    evidence_refs=[h.source for h in reg.conflicting_hypotheses],
                )

                if reg.verification_state in ("UNCERTAIN", "CONFLICTED") or reg.confidence < 0.65:
                    uncertain_question_candidates.append(q_obj)

                extracted_questions.append(q_obj)
                if curr_section_id and curr_section_id in sections_by_id:
                    sections_by_id[curr_section_id].question_ids.append(q_id)

                curr_question = q_obj
                order_counter += 1
                continue

            # --- Main Question Candidate e.g. Q1. or 2. ---
            if main_m:
                q_num = main_m.group(1)
                q_text = main_m.group(2).strip() or txt
                q_id = f"{document_id}:{reg.region_id}"
                disp_num = q_num

                q_type = "SHORT_ANSWER"
                if len(q_text) > 120 or "explain" in q_text.lower() or "discuss" in q_text.lower():
                    q_type = "LONG_ANSWER"

                q_obj = Question(
                    id=q_id,
                    number=disp_num,
                    text=txt,  # 100% original OCR text preserved
                    page=reg.page,
                    bbox=reg.bbox,
                    order_index=order_counter,
                    section=curr_section_title,
                    section_id=curr_section_id,
                    section_title=curr_section_title,
                    parent_question_id=None,
                    question_type=q_type,
                    source_region_ids=[reg.region_id],
                    source_regions=[Region(page=reg.page, bbox=reg.bbox)],
                    extraction_confidence=reg.confidence,
                    verification_state=reg.verification_state,
                    evidence_refs=[h.source for h in reg.conflicting_hypotheses],
                )

                if reg.verification_state in ("UNCERTAIN", "CONFLICTED") or reg.confidence < 0.65:
                    uncertain_question_candidates.append(q_obj)

                extracted_questions.append(q_obj)
                if curr_section_id and curr_section_id in sections_by_id:
                    sections_by_id[curr_section_id].question_ids.append(q_id)

                curr_question = q_obj
                current_main_parent_id = q_id
                order_counter += 1
                continue


            # --- Continuation Region (Multi-line or Multi-page continuation) ---
            if curr_question is not None:
                # Append exact OCR text without modifying original words
                curr_question.text = f"{curr_question.text} {txt}"
                curr_question.source_region_ids.append(reg.region_id)
                curr_question.source_regions.append(Region(page=reg.page, bbox=reg.bbox))
                curr_question.bbox = _compute_bounding_box(
                    [DocumentRegion(region_id=r_id, page=r.page, bbox=r.bbox, text="") for r_id, r in zip(curr_question.source_region_ids, curr_question.source_regions)]
                )
                if curr_question.page != reg.page:
                    audit.multi_page_question_count += 1
                else:
                    audit.multi_region_question_count += 1

        # ---------------------------------------------------------------------
        # Stage 4: MCQ Structure Extraction with Authoritative Region Storage
        # ---------------------------------------------------------------------
        for opt_reg in option_regions:
            opt_m = OPTION_PREFIX_RE.match(opt_reg.text.strip())
            label = opt_m.group(1).upper() if opt_m else "A"
            text_val = opt_m.group(2).strip() if opt_m else opt_reg.text.strip()

            opt_x_center = opt_reg.bbox.x + opt_reg.bbox.width / 2.0
            p_w = 1000.0
            if doc_understanding_result and doc_understanding_result.pages:
                p_match = next((p for p in doc_understanding_result.pages if p.page_number == opt_reg.page), None)
                if p_match and p_match.width > 0:
                    p_w = p_match.width

            # Find matching parent question using spatial column compatibility & preceding layout position
            candidate_matches = []
            for q in extracted_questions:
                if q.page != opt_reg.page and q.page != opt_reg.page - 1:
                    continue

                q_x_center = q.bbox.x + q.bbox.width / 2.0
                col_compatible = abs(opt_x_center - q_x_center) < (0.35 * p_w) or (opt_reg.bbox.x >= q.bbox.x - 50 and opt_reg.bbox.x <= q.bbox.x + q.bbox.width + 50)
                is_preceding = (opt_reg.page > q.page) or (opt_reg.page == q.page and opt_reg.bbox.y >= q.bbox.y - 15.0)

                if col_compatible and is_preceding:
                    page_diff = opt_reg.page - q.page
                    y_diff = opt_reg.bbox.y - q.bbox.y if page_diff == 0 else opt_reg.bbox.y
                    candidate_matches.append((page_diff, y_diff, q))

            target_q = None
            if candidate_matches:
                candidate_matches.sort(key=lambda item: (item[0], item[1]))
                target_q = candidate_matches[0][2]
            else:
                fallback_candidates = [q for q in extracted_questions if q.page == opt_reg.page and opt_reg.bbox.y >= q.bbox.y - 15.0]
                if fallback_candidates:
                    fallback_candidates.sort(key=lambda q: opt_reg.bbox.y - q.bbox.y)
                    target_q = fallback_candidates[0]

            if target_q is not None:
                opt_id = f"opt_{target_q.id}_{label}_{uuid.uuid4().hex[:4]}"
                extracted_opt = ExtractedOption(
                    option_id=opt_id,
                    question_id=target_q.id,
                    label=label,
                    text=opt_reg.text.strip(),  # Authoritative exact OCR text
                    source_region_ids=[opt_reg.region_id],
                    source_regions=[Region(page=opt_reg.page, bbox=opt_reg.bbox)],
                    extraction_confidence=opt_reg.confidence,
                    verification_state=opt_reg.verification_state,
                )
                target_q.extracted_options.append(extracted_opt)
                target_q.options.append(f"{label}. {text_val}")  # Backward compatible display array
                target_q.question_type = "MCQ"

        # Update final audit stats & multi-region / multi-page counts from extracted questions
        audit.accepted_question_count = len(extracted_questions)
        audit.rejection_reasons = rejection_records

        multi_p_count = 0
        multi_r_count = 0
        for q in extracted_questions:
            q_pages = {r.page for r in q.source_regions}
            if len(q_pages) > 1:
                multi_p_count += 1
            if len(q.source_region_ids) > 1:
                multi_r_count += 1

        audit.multi_page_question_count = multi_p_count
        audit.multi_region_question_count = multi_r_count

        # Build DocumentStructureGraph
        struct_graph = self.doc_understanding_service.build_structure_graph(doc_understanding_result)

        # Invariant Verification: Check for internal graph/audit contradictions
        cross_page_rels = [
            rel for rel in doc_understanding_result.relationships if rel.relationship_type == "continuation_of"
        ]
        has_cross_page_edges = any(
            next((r.page for r in doc_understanding_result.regions if r.region_id == rel.source_region_id), None) !=
            next((r.page for r in doc_understanding_result.regions if r.region_id == rel.target_region_id), None)
            for rel in cross_page_rels
        )

        if has_cross_page_edges and audit.multi_page_question_count == 0:
            audit.invariant_violations.append("INVARIANT_VIOLATION: Cross-page continuations exist in graph but multi_page_question_count is 0")

        duplicate_ids = len(extracted_questions) - len({q.id for q in extracted_questions})
        if duplicate_ids > 0:
            audit.invariant_violations.append(f"INVARIANT_VIOLATION: {duplicate_ids} duplicate internal question IDs detected")

        return DocumentQuestionExtractionResult(
            document_id=document_id,
            questions=extracted_questions,
            sections=classified_sections,
            uncertain_candidates=uncertain_question_candidates,
            audit=audit,
            structure_graph=struct_graph,
            fallback_used=False,
            invariant_violations=audit.invariant_violations,
        )

