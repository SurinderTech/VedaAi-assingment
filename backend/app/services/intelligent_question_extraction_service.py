"""
Intelligent Question Extraction Service — Graph-Driven Architecture.

The DocumentStructureGraph drives question extraction.
Questions are found by walking graph nodes with role=QUESTION,
not by regex pattern matching.

Architecture:
  DocumentStructureGraph (built by DocumentUnderstandingService)
    → Find QUESTION nodes
    → Follow edges for OPTIONS, SUBQUESTIONS, CONTINUATIONS
    → Assemble text from source regions (zero hallucination)
    → Fall back to regex ONLY when graph is empty/insufficient

Core Principles:
1. Graph-first: The graph is the primary decision-making input
2. Zero-hallucination: Every character comes from OCR source regions
3. VLM independence: Works when VLM disabled (graph from deterministic evidence)
4. Safe fallback: Falls back to regex when graph is truly empty
5. Audit trail: Full diagnostic metadata for every extraction decision
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
    DocumentStructureGraph,
    GraphNode,
    GraphEdge,
    StructureHypothesis,
    VerificationState,
)
from app.services.document_understanding_service import DocumentUnderstandingService


# --- Regex patterns for legacy fallback ---
OPTION_PREFIX_RE = re.compile(
    r"^\s*[\(\[]?\s*([A-Da-d1-4])\s*[\)\]\.\:]\s*(.*)$"
)

SUBQUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*)?\d{1,3}\s*[\.\:\-]?\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)$",
    re.IGNORECASE,
)

MAIN_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?\d{1,3}\s*[\.\):\-]\s*(.*)$",
    re.IGNORECASE,
)

SECTION_HEADER_RE = re.compile(
    r"^\s*(?:SECTION|PART|GROUP)\s*[\-\–\:\s]*([A-Z0-9]{1,3})\s*$",
    re.IGNORECASE,
)


def _compute_bounding_box(regions: List[DocumentRegion]) -> Optional[BBox]:
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
    column_map: Dict[str, int] = {}
    if not regions or page_width <= 0:
        for r in regions:
            column_map[r.region_id] = 0
        return column_map

    mid = page_width / 2.0
    left_count = sum(1 for r in regions if (r.bbox.x + r.bbox.width / 2.0) < mid and r.bbox.width < page_width * 0.6)
    right_count = sum(1 for r in regions if (r.bbox.x + r.bbox.width / 2.0) >= mid and r.bbox.width < page_width * 0.6)
    is_multi_column = (left_count >= 2 and right_count >= 2)

    for r in regions:
        if is_multi_column:
            x_center = r.bbox.x + r.bbox.width / 2.0
            column_map[r.region_id] = 0 if x_center < mid else 1
        else:
            column_map[r.region_id] = 0
    return column_map


class IntelligentQuestionExtractionService:
    """
    Graph-driven question extraction service.

    Primary path: Walk the DocumentStructureGraph to extract questions.
    Fallback path: Regex-based extraction when graph is empty.
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
        Extracts questions using graph-driven approach.

        1. If structure_graph exists with QUESTION nodes → walk graph
        2. Else → fall back to regex-based extraction
        """
        if not blocks:
            return DocumentQuestionExtractionResult(
                document_id=document_id,
                questions=[],
                sections=[],
                uncertain_candidates=[],
                audit=ExtractionAudit(),
            )

        # Ensure DocumentUnderstandingResult exists
        if doc_understanding_result is None or not doc_understanding_result.regions:
            doc_understanding_result = self.doc_understanding_service.process_document(
                blocks=blocks, document_id=document_id, page_sizes=page_sizes
            )

        # PRIMARY PATH: Graph-driven extraction
        graph = doc_understanding_result.structure_graph
        if graph and graph.nodes:
            question_nodes = [n for n in graph.nodes.values() if n.role == "QUESTION"]
            if question_nodes:
                print(f"[IntelligentExtraction] Graph-driven extraction: {len(question_nodes)} QUESTION nodes found")
                return self._extract_from_graph(
                    graph=graph,
                    doc_result=doc_understanding_result,
                    document_id=document_id,
                )

        # FALLBACK: Regex-based extraction from regions
        print("[IntelligentExtraction] Graph has no QUESTION nodes, using region-based fallback")
        return self._extract_from_regions(
            doc_result=doc_understanding_result,
            document_id=document_id,
        )

    # ================================================================
    # PRIMARY: Graph-Driven Extraction
    # ================================================================

    def _extract_from_graph(
        self,
        graph: DocumentStructureGraph,
        doc_result: DocumentUnderstandingResult,
        document_id: str,
    ) -> DocumentQuestionExtractionResult:
        """
        Walks the DocumentStructureGraph to construct questions.

        QUESTION nodes → questions
        OPTION edges → MCQ options attached to parent question
        SUBQUESTION edges → subquestions
        CONTINUATION edges → multi-line/multi-page question text
        SECTION_HEADER nodes → section containers
        """
        region_map = {r.region_id: r for r in doc_result.regions}
        audit = ExtractionAudit(candidate_count=len(graph.nodes))

        if graph.graph_semantic_state in ("AMBIGUOUS", "UNRESOLVED", "CONFLICTING"):
            for node in graph.nodes.values():
                if node.semantic_state in ("AMBIGUOUS", "UNRESOLVED", "CONFLICTING"):
                    audit.uncertain_count += 1
                    audit.conflicts.append(f"Graph semantic state is {graph.graph_semantic_state} for node {node.region_id}.")
            return DocumentQuestionExtractionResult(
                document_id=document_id,
                questions=[],
                sections=[],
                uncertain_candidates=[],
                audit=audit,
                structure_graph=graph,
                fallback_used=False,
                invariant_violations=[f"Semantic validation blocked extraction: graph semantic state is {graph.graph_semantic_state}"],
            )

        # Build edge indices
        # target_id → list of (source_id, relationship, confidence)
        children_of: Dict[str, List[Tuple[str, str, float]]] = {}
        # source_id → list of (target_id, relationship, confidence)
        parents_of: Dict[str, List[Tuple[str, str, float]]] = {}

        for edge in graph.edges:
            children_of.setdefault(edge.target_id, []).append(
                (edge.source_id, edge.relationship, edge.confidence)
            )
            parents_of.setdefault(edge.source_id, []).append(
                (edge.target_id, edge.relationship, edge.confidence)
            )

        # Find sections
        sections: List[ExtractedSection] = []
        section_for_region: Dict[str, str] = {}  # region_id → section_title

        for node_id, node in graph.nodes.items():
            if node.role == "SECTION_HEADER":
                sec_title = node.text.strip()
                sec_m = SECTION_HEADER_RE.match(sec_title)
                if sec_m:
                    sec_title = f"Section-{sec_m.group(1).upper()}"
                sec_id = f"sec_{len(sections)+1}_{sec_title[:20].lower().replace(' ','_')}"
                sec = ExtractedSection(
                    section_id=sec_id,
                    title=sec_title,
                    page=node.page,
                    bbox=node.bbox,
                    source_region_ids=[node_id],
                )
                sections.append(sec)
                audit.section_count += 1

                # Find section members via edges
                for child_id, rel_type, _ in children_of.get(node_id, []):
                    if rel_type == "section_member":
                        section_for_region[child_id] = sec_title

                audit.rejected_count += 1

        # Find QUESTION nodes and build questions
        question_nodes = sorted(
            [n for n in graph.nodes.values() if n.role == "QUESTION"],
            key=lambda n: (n.page, n.bbox.y, n.bbox.x),
        )

        extracted_questions: List[Question] = []
        uncertain_candidates: List[Question] = []
        rejection_records: List[RejectionRecord] = []
        order_counter = 0

        for q_node in question_nodes:
            if q_node.semantic_state in ("AMBIGUOUS", "UNRESOLVED", "CONFLICTING"):
                uncertain_candidates.append(Question(
                    id=f"{document_id}:{q_node.region_id}",
                    number=self._extract_display_number(q_node.text.strip()),
                    text=q_node.text.strip(),
                    page=q_node.page,
                    bbox=q_node.bbox,
                    source_region_ids=[q_node.region_id],
                    source_regions=[Region(page=q_node.page, bbox=q_node.bbox)],
                    extraction_confidence=q_node.confidence,
                    verification_state="UNCERTAIN",
                ))
                audit.uncertain_count += 1
                continue

            region = region_map.get(q_node.region_id)
            if not region:
                continue

            # Extract display number from text
            q_text = q_node.text.strip()
            display_num = self._extract_display_number(q_text)

            # Build question ID: document_id:region_id
            q_id = f"Q{self._extract_display_number(q_text)}"

            # Find section for this question
            sec_title = section_for_region.get(q_node.region_id)

            # Collect continuation text
            continuation_ids, continuation_text = self._collect_continuations(
                q_node.region_id, children_of, graph.nodes, region_map
            )

            # Assemble full question text from source regions
            full_text = q_text
            all_source_ids = [q_node.region_id]
            all_source_regions = [Region(page=q_node.page, bbox=q_node.bbox)]

            for cont_id in continuation_ids:
                if cont_id not in all_source_ids:
                    cont_node = graph.nodes.get(cont_id)
                    if cont_node:
                        full_text = f"{full_text} {cont_node.text.strip()}"
                        all_source_ids.append(cont_id)
                        all_source_regions.append(Region(page=cont_node.page, bbox=cont_node.bbox))

            # Determine question type
            q_type = "SHORT_ANSWER"
            if len(full_text) > 120 or "explain" in full_text.lower() or "discuss" in full_text.lower():
                q_type = "LONG_ANSWER"

            q_obj = Question(
                id=q_id,
                number=display_num,
                text=full_text,
                page=q_node.page,
                bbox=_compute_bounding_box([region_map[rid] for rid in all_source_ids if rid in region_map]) or q_node.bbox,
                order_index=order_counter,
                section=sec_title,
                parent_question_id=(
                    f"Q{display_num.split('(', 1)[0]}"
                    if "(" in display_num else None
                ),
                question_type=q_type,
                source_region_ids=all_source_ids,
                source_regions=all_source_regions,
                extraction_confidence=q_node.confidence,
                verification_state=region.verification_state if region else "UNVERIFIED",
                evidence_refs=[h.source for h in region.conflicting_hypotheses] if region else [],
            )

            # Find and attach OPTIONS
            option_count = self._attach_options(
                question=q_obj,
                question_node_id=q_node.region_id,
                children_of=children_of,
                graph_nodes=graph.nodes,
                region_map=region_map,
            )
            if option_count > 0:
                q_obj.question_type = "MCQ"

            # Find and attach SUBQUESTIONS
            self._attach_subquestions(
                parent_question=q_obj,
                question_node_id=q_node.region_id,
                children_of=children_of,
                graph_nodes=graph.nodes,
                region_map=region_map,
                document_id=document_id,
                extracted_questions=extracted_questions,
                order_counter_ref=[order_counter + 1],
                sec_title=sec_title,
            )

            # Check if uncertain
            if region and (region.verification_state in ("UNCERTAIN", "CONFLICTED") or region.confidence < 0.65):
                uncertain_candidates.append(q_obj)

            extracted_questions.append(q_obj)
            order_counter += 1

        # Record non-question rejections for audit
        for node_id, node in graph.nodes.items():
            if node.role in ("HEADER", "FOOTER", "METADATA", "INSTRUCTION", "SECTION_HEADER"):
                reason_label = "administrative" if node.role in ("HEADER", "FOOTER", "METADATA", "INSTRUCTION") else "section"
                rejection_records.append(RejectionRecord(
                    region_id=node_id,
                    ocr_text=node.text[:60],
                    classification=node.role,
                    confidence=node.confidence,
                    reason=f"Administrative {node.role.lower()} content — not a question.",
                ))
                audit.rejected_count += 1

        audit.accepted_question_count = len(extracted_questions)
        audit.rejection_reasons = rejection_records

        # Count multi-region / multi-page questions and validate graph invariants
        edge_map = {(e.source_id, e.target_id): e.relationship for e in graph.edges}
        question_id_to_region_id = {q.id: q.source_region_ids[0] for q in extracted_questions if q.source_region_ids}

        for q in extracted_questions:
            q_pages = {r.page for r in q.source_regions}
            if len(q_pages) > 1:
                audit.multi_page_question_count += 1
            if len(q.source_region_ids) > 1:
                audit.multi_region_question_count += 1

            # Invariant 1: Every source_region_id must resolve to an existing graph node
            for rid in q.source_region_ids:
                if rid not in graph.nodes:
                    err = f"Invariant Violation: Question {q.id} source_region_id '{rid}' not found in DocumentStructureGraph."
                    audit.invariant_violations.append(err)

            # Invariant 2: Subquestion parent_question_id must resolve to an actual parent QUESTION node with subquestion_of edge
            if q.parent_question_id:
                parent_region_id = question_id_to_region_id.get(q.parent_question_id)
                if not parent_region_id or parent_region_id not in graph.nodes:
                    err = f"Invariant Violation: Subquestion {q.id} parent_question_id '{q.parent_question_id}' does not resolve to an existing graph node."
                    audit.invariant_violations.append(err)
                else:
                    parent_node = graph.nodes[parent_region_id]
                    if parent_node.role != "QUESTION":
                        err = f"Invariant Violation: Subquestion {q.id} parent node '{parent_region_id}' has role '{parent_node.role}', expected 'QUESTION'."
                        audit.invariant_violations.append(err)

                    sub_region_id = q.source_region_ids[0] if q.source_region_ids else None
                    if sub_region_id and edge_map.get((sub_region_id, parent_region_id)) != "subquestion_of":
                        err = f"Invariant Violation: Subquestion {q.id} ('{sub_region_id}') missing 'subquestion_of' edge to parent node '{parent_region_id}'."
                        audit.invariant_violations.append(err)

        return DocumentQuestionExtractionResult(
            document_id=document_id,
            questions=extracted_questions,
            sections=sections,
            uncertain_candidates=uncertain_candidates,
            audit=audit,
            structure_graph=graph,
            fallback_used=False,
            invariant_violations=audit.invariant_violations,
        )

    def _extract_display_number(self, text: str, active_parent_num: str = "1") -> str:
        """Extracts the display question number from text like 'Q1. Explain...' -> '1' or 'a) Define...' -> '1(a)'."""
        # Subquestion pattern with main digit: 1(a) or Q1(a)
        m_sub = re.match(
            r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\:\-]?\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]",
            text, re.IGNORECASE
        )
        if m_sub:
            return f"{m_sub.group(1)}({m_sub.group(2).lower()})"

        # Standalone subquestion letter: a) or (a) or a.
        m_let = re.match(
            r"^\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]",
            text, re.IGNORECASE
        )
        if m_let:
            return f"{active_parent_num}({m_let.group(1).lower()})"

        # Primary main question number: Q1. or 1. or 1)
        m = re.match(
            r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})\s*[\.\):\-]",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1)

        return str(uuid.uuid4())[:6]

    def _collect_continuations(
        self,
        node_id: str,
        children_of: Dict[str, List[Tuple[str, str, float]]],
        graph_nodes: Dict[str, GraphNode],
        region_map: Dict[str, DocumentRegion],
    ) -> Tuple[List[str], str]:
        """Collects continuation regions for a question node."""
        continuation_ids: List[str] = []
        continuation_text = ""

        # Find all regions that are continuation_of this node
        for child_id, rel_type, conf in children_of.get(node_id, []):
            if rel_type == "continuation_of":
                child_node = graph_nodes.get(child_id)
                if child_node and child_node.role not in ("QUESTION", "SECTION_HEADER", "OPTION"):
                    continuation_ids.append(child_id)
                    continuation_text += " " + child_node.text.strip()

        # Sort continuations by spatial order
        continuation_ids.sort(key=lambda cid: (
            graph_nodes[cid].page if cid in graph_nodes else 0,
            graph_nodes[cid].bbox.y if cid in graph_nodes else 0,
        ))

        return continuation_ids, continuation_text

    def _attach_options(
        self,
        question: Question,
        question_node_id: str,
        children_of: Dict[str, List[Tuple[str, str, float]]],
        graph_nodes: Dict[str, GraphNode],
        region_map: Dict[str, DocumentRegion],
    ) -> int:
        """Attaches MCQ options to a question from graph edges."""
        option_count = 0

        for child_id, rel_type, conf in children_of.get(question_node_id, []):
            if rel_type != "option_of":
                continue

            child_node = graph_nodes.get(child_id)
            if not child_node or child_node.role != "OPTION":
                continue

            region = region_map.get(child_id)
            opt_text = child_node.text.strip()

            # Extract option label
            opt_m = OPTION_PREFIX_RE.match(opt_text)
            label = opt_m.group(1).upper() if opt_m else chr(65 + option_count)  # A, B, C, D
            text_val = opt_m.group(2).strip() if opt_m else opt_text

            opt_id = f"opt_{question.id}_{label}_{uuid.uuid4().hex[:4]}"
            extracted_opt = ExtractedOption(
                option_id=opt_id,
                question_id=question.id,
                label=label,
                text=opt_text,  # Full OCR text preserved
                source_region_ids=[child_id],
                source_regions=[Region(page=child_node.page, bbox=child_node.bbox)],
                extraction_confidence=child_node.confidence,
                verification_state=region.verification_state if region else "UNVERIFIED",
            )
            question.extracted_options.append(extracted_opt)
            question.options.append(f"{label}. {text_val}")
            option_count += 1
            audit_count = getattr(question, '_opt_audit_count', 0)

        return option_count

    def _attach_subquestions(
        self,
        parent_question: Question,
        question_node_id: str,
        children_of: Dict[str, List[Tuple[str, str, float]]],
        graph_nodes: Dict[str, GraphNode],
        region_map: Dict[str, DocumentRegion],
        document_id: str,
        extracted_questions: List[Question],
        order_counter_ref: List[int],
        sec_title: Optional[str],
    ) -> None:
        """Attaches subquestions to a parent question from graph edges."""
        for child_id, rel_type, conf in children_of.get(question_node_id, []):
            if rel_type != "subquestion_of":
                continue

            child_node = graph_nodes.get(child_id)
            if not child_node or child_node.role != "SUBQUESTION":
                continue

            region = region_map.get(child_id)
            sub_text = child_node.text.strip()

            # Extract subquestion display number
            m = re.match(r"^\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)", sub_text, re.IGNORECASE)
            sub_label = m.group(1).lower() if m else "?"
            parent_num = parent_question.number
            display_num = f"{parent_num}({sub_label})"

            sub_id = f"Q{display_num.replace(' ', '')}"
            sub_q = Question(
                id=sub_id,
                number=display_num,
                text=sub_text,
                page=child_node.page,
                bbox=child_node.bbox,
                order_index=order_counter_ref[0],
                section=sec_title,
                parent_question_id=parent_question.id,
                question_type="SUBQUESTION",
                source_region_ids=[child_id],
                source_regions=[Region(page=child_node.page, bbox=child_node.bbox)],
                extraction_confidence=child_node.confidence,
                verification_state=region.verification_state if region else "UNVERIFIED",
            )
            extracted_questions.append(sub_q)
            order_counter_ref[0] += 1

    # ================================================================
    # FALLBACK: Region-based extraction (when graph has no QUESTION nodes)
    # ================================================================

    def _extract_from_regions(
        self,
        doc_result: DocumentUnderstandingResult,
        document_id: str,
    ) -> DocumentQuestionExtractionResult:
        """
        Fallback extraction using regex on regions.
        Used ONLY when the structure graph has no QUESTION nodes.
        """
        regions = doc_result.regions
        audit = ExtractionAudit(candidate_count=len(regions))
        rejection_records: List[RejectionRecord] = []
        classified_sections: List[ExtractedSection] = []
        extracted_questions: List[Question] = []
        uncertain_candidates: List[Question] = []
        option_regions: List[DocumentRegion] = []

        active_section: Optional[ExtractedSection] = None
        sections_by_id: Dict[str, ExtractedSection] = {}
        accepted_regions: List[DocumentRegion] = []
        curr_section_title: Optional[str] = None

        for reg in regions:
            text_low = reg.text.strip().lower()

            # Section header
            sec_m = SECTION_HEADER_RE.match(reg.text.strip())
            if sec_m or reg.region_type == "SECTION_HEADER":
                sec_title = f"Section-{sec_m.group(1).upper()}" if sec_m else reg.text.strip()
                sec_id = f"sec_{len(classified_sections)+1}"
                active_section = ExtractedSection(
                    section_id=sec_id, title=sec_title, page=reg.page,
                    bbox=reg.bbox, source_region_ids=[reg.region_id],
                )
                classified_sections.append(active_section)
                sections_by_id[sec_id] = active_section
                curr_section_title = sec_title
                audit.section_count += 1
                continue

            # Administrative/instruction filtering
            if reg.region_type in ("HEADER", "FOOTER", "METADATA", "INSTRUCTION"):
                audit.rejected_count += 1
                rejection_records.append(RejectionRecord(
                    region_id=reg.region_id, ocr_text=reg.text[:60],
                    classification=reg.region_type, confidence=reg.confidence,
                    reason=f"Non-question content ({reg.region_type}).",
                ))
                continue

            # Option
            opt_m = OPTION_PREFIX_RE.match(reg.text.strip())
            if reg.region_type == "OPTION" or (opt_m and not MAIN_QUESTION_PREFIX_RE.match(reg.text.strip())):
                option_regions.append(reg)
                audit.option_count += 1
                continue

            accepted_regions.append(reg)

        # Build questions from accepted regions using regex
        curr_question: Optional[Question] = None
        order_counter = 0

        # Sort by reading order
        regions_by_page: Dict[int, List[DocumentRegion]] = {}
        for r in accepted_regions:
            regions_by_page.setdefault(r.page, []).append(r)

        ordered_regions: List[DocumentRegion] = []
        for page_num in sorted(regions_by_page.keys()):
            p_regs = regions_by_page[page_num]
            p_width = 1000.0
            if doc_result.pages:
                p_match = next((p for p in doc_result.pages if p.page_number == page_num), None)
                if p_match and p_match.width > 0:
                    p_width = p_match.width
            col_map = _detect_reading_columns(p_regs, p_width)
            sorted_regs = sorted(p_regs, key=lambda r: (col_map.get(r.region_id, 0), r.bbox.y, r.bbox.x))
            ordered_regions.extend(sorted_regs)

        for reg in ordered_regions:
            txt = reg.text.strip()
            if not txt:
                continue

            combined_sub_m = re.match(
                r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\:\-]?\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.:]\s*(.*)$",
                txt,
                re.IGNORECASE,
            )
            if combined_sub_m:
                main_num = combined_sub_m.group(1)
                sub_label = combined_sub_m.group(2).lower()
                body = combined_sub_m.group(3).strip()
                q_id = f"Q{main_num}({sub_label})"

                q_obj = Question(
                    id=q_id,
                    number=f"{main_num}({sub_label})",
                    text=txt,
                    page=reg.page,
                    bbox=reg.bbox,
                    order_index=order_counter,
                    section=curr_section_title,
                    parent_question_id=f"Q{main_num}",
                    question_type="SUBQUESTION",
                    source_region_ids=[reg.region_id],
                    source_regions=[Region(page=reg.page, bbox=reg.bbox)],
                    extraction_confidence=reg.confidence,
                    verification_state=reg.verification_state,
                )
                extracted_questions.append(q_obj)
                if reg.verification_state in ("UNCERTAIN", "CONFLICTED") or reg.confidence < 0.65:
                    uncertain_candidates.append(q_obj)
                    audit.uncertain_count += 1
                order_counter += 1
                continue

            main_m = MAIN_QUESTION_PREFIX_RE.match(txt)
            if main_m:
                q_num_m = re.match(r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})", txt, re.IGNORECASE)
                q_num = q_num_m.group(1) if q_num_m else str(order_counter + 1)
                q_id = f"{document_id}:{reg.region_id}"

                curr_question = Question(
                    id=q_id, number=q_num, text=txt, page=reg.page,
                    bbox=reg.bbox, order_index=order_counter,
                    section=curr_section_title,
                    source_region_ids=[reg.region_id],
                    source_regions=[Region(page=reg.page, bbox=reg.bbox)],
                    extraction_confidence=reg.confidence,
                    verification_state=reg.verification_state,
                )
                extracted_questions.append(curr_question)
                if reg.verification_state in ("UNCERTAIN", "CONFLICTED") or reg.confidence < 0.65:
                    uncertain_candidates.append(curr_question)
                    audit.uncertain_count += 1
                order_counter += 1
                continue

            # Continuation
            if curr_question is not None:
                curr_question.text = f"{curr_question.text} {txt}"
                curr_question.source_region_ids.append(reg.region_id)
                curr_question.source_regions.append(Region(page=reg.page, bbox=reg.bbox))

        # Attach options to questions
        for opt_reg in option_regions:
            opt_m = OPTION_PREFIX_RE.match(opt_reg.text.strip())
            label = opt_m.group(1).upper() if opt_m else "A"
            text_val = opt_m.group(2).strip() if opt_m else opt_reg.text.strip()

            target_q = None
            for q in reversed(extracted_questions):
                if q.page == opt_reg.page and opt_reg.bbox.y >= q.bbox.y - 15.0:
                    target_q = q
                    break

            if target_q:
                opt_id = f"opt_{target_q.id}_{label}_{uuid.uuid4().hex[:4]}"
                target_q.extracted_options.append(ExtractedOption(
                    option_id=opt_id, question_id=target_q.id, label=label,
                    text=opt_reg.text.strip(),
                    source_region_ids=[opt_reg.region_id],
                    source_regions=[Region(page=opt_reg.page, bbox=opt_reg.bbox)],
                    extraction_confidence=opt_reg.confidence,
                ))
                target_q.options.append(f"{label}. {text_val}")
                target_q.question_type = "MCQ"

        audit.accepted_question_count = len(extracted_questions)
        audit.rejection_reasons = rejection_records

        graph = doc_result.structure_graph or self.doc_understanding_service.build_structure_graph(doc_result)

        return DocumentQuestionExtractionResult(
            document_id=document_id,
            questions=extracted_questions,
            sections=classified_sections,
            uncertain_candidates=uncertain_candidates,
            audit=audit,
            structure_graph=graph,
            fallback_used=True,  # Explicitly mark that regex fallback was used
            invariant_violations=audit.invariant_violations,
        )
