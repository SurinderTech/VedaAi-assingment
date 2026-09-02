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
        # Spatial index: page → sorted list of DocumentRegions (for option body fallback)
        page_regions_by_page: Dict[int, List[DocumentRegion]] = {}
        for r in doc_result.regions:
            page_regions_by_page.setdefault(r.page, []).append(r)
        for pg in page_regions_by_page:
            page_regions_by_page[pg].sort(key=lambda r: (r.bbox.y, r.bbox.x))
        audit = ExtractionAudit(candidate_count=len(graph.nodes))

        if graph.graph_semantic_state in ("AMBIGUOUS", "UNRESOLVED", "CONFLICTING"):
            # FIX 3: Only block if there are truly no QUESTION nodes.
            # If VLM has produced QUESTION nodes, extract them even if some
            # edges are unresolved — per-node state handles individual confidence.
            question_nodes_check = [n for n in graph.nodes.values() if n.role == "QUESTION"]
            if not question_nodes_check:
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
                    invariant_violations=[f"Semantic validation blocked extraction: graph semantic state is {graph.graph_semantic_state} and no QUESTION nodes found."],
                )
            else:
                print(
                    f"[IntelligentExtraction] FIX-3: Graph state is {graph.graph_semantic_state} "
                    f"but {len(question_nodes_check)} QUESTION node(s) found — proceeding with extraction."
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

        # Fix 2: Document-scoped deduplication
        # Track seen display_numbers and (page, y_approx) to reject exact duplicates
        seen_display_numbers: Dict[str, Tuple[str, float]] = {}  # display_num → (region_id, confidence)
        seen_spatial_keys: set = set()  # (page, round(y/10)*10) for spatial dedup
        rejected_region_ids: set = set()  # region_ids retroactively rejected by higher-conf duplicate

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

            # --- FRAGMENT GUARD: Reject roman-numeral / letter-only subquestion markers ---
            # Patterns like (i), (ii), (iii), (iv), [i], [a], [b] that appear as
            # standalone QUESTION nodes are subquestion index markers, not real questions.
            # They have no question body and must not be emitted as separate questions.
            _fragment_re = re.compile(
                r'^\s*[\(\[]?\s*(?:i{1,4}v?|vi{0,3}|ix|x|[a-d]|[ivxlcdm]{1,4})\s*[\)\]\.:]?\s*$',
                re.IGNORECASE
            )
            if _fragment_re.match(q_text) and len(q_text.strip()) <= 8:
                audit.rejected_count += 1
                print(
                    f"[IntelligentExtraction] Fragment guard: rejected lone subquestion marker "
                    f"'{q_text}' (region={q_node.region_id}) — not a real question."
                )
                continue

            # Also reject pure number/label nodes with no body text
            # e.g. node text is literally "1" or "Q2" with nothing after
            _bare_label_re = re.compile(r'^\s*(?:Q\.?)?\d{1,3}\.?\s*$', re.IGNORECASE)
            if _bare_label_re.match(q_text):
                audit.rejected_count += 1
                print(
                    f"[IntelligentExtraction] Fragment guard: rejected bare question label "
                    f"'{q_text}' (region={q_node.region_id}) — label-only node."
                )
                continue

            # --- GROUP-PARENT GUARD ---
            # Some nodes the VLM may (or may not) label QUESTION are actually
            # group-parent intro headers — they introduce sub-questions but are
            # not themselves answerable.  Detection criteria (document-agnostic):
            #   • Text ends with ":" after stripping trailing punctuation/spaces,  OR
            #   • Body after the question number contains only a short intro phrase
            #     whose entire content is a known-structural imperative (no real
            #     question content follows).
            # This guard is purely structural/linguistic and relies on no document-
            # specific keywords, coordinates, or templates.
            _group_parent_intro_re = re.compile(
                r'\b(?:write\s+briefly|answer\s+the\s+following|attempt\s+any|'
                r'short\s+answer\s+(?:questions?)?|attempt\s+the\s+following|'
                r'answer\s+any|describe\s+briefly|explain\s+briefly|'
                r'give\s+(?:short\s+)?(?:answers?|notes?))\b',
                re.IGNORECASE,
            )
            # Strip number prefix to get the body ("1. Write briefly :" → "Write briefly :")
            _body_after_num = re.sub(
                r'^\s*(?:Q(?:uestion)?\.?\s*)?\d{1,3}\s*[.):–-]?\s*', '', q_text
            ).strip()
            _ends_with_colon = q_text.rstrip().endswith(':')
            _is_intro_only = bool(
                _group_parent_intro_re.search(_body_after_num)
                and len(_body_after_num) < 80       # intro phrase is short
            )
            if _ends_with_colon or _is_intro_only:
                audit.rejected_count += 1
                print(
                    f"[IntelligentExtraction] Group-parent guard: rejected intro-only node "
                    f"'{q_text[:60]}' (region={q_node.region_id}) — promoting its subquestions to top-level."
                )
                # PROMOTION: Walk subquestion_of children and emit them as top-level questions.
                # Without this, Section-A style "1. Write briefly : → (a)(b)…(j)" would lose
                # all sub-questions when the parent is rejected, making student answers unmatchable.
                _parent_num = display_num or re.sub(
                    r'^\s*(?:Q(?:uestion)?\.?\s*)?\D*(\d{1,3})\s*.*', r'\1', q_text
                ).strip() or "1"
                for _child_id, _rel, _conf in children_of.get(q_node.region_id, []):
                    if _rel != "subquestion_of":
                        continue
                    _child_node = graph.nodes.get(_child_id)
                    if not _child_node or _child_node.role not in ("SUBQUESTION", "QUESTION", "UNKNOWN"):
                        continue
                    _child_text = _child_node.text.strip()
                    if not _child_text:
                        continue
                    # Derive display number: "(a) Define..." → "1(a)"
                    _sub_m = re.match(
                        r'^\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)', _child_text, re.IGNORECASE
                    )
                    _sub_label = _sub_m.group(1).lower() if _sub_m else None
                    _promoted_num = f"{_parent_num}({_sub_label})" if _sub_label else _child_text[:10]
                    _promoted_id = f"Q{_promoted_num.replace(' ', '')}"

                    # Skip duplicates
                    if _promoted_num in seen_display_numbers:
                        continue
                    _sp_key = (_child_node.page, round(_child_node.bbox.y / 5.0))
                    if _sp_key in seen_spatial_keys:
                        continue

                    seen_display_numbers[_promoted_num] = (_child_id, _child_node.confidence)
                    seen_spatial_keys.add(_sp_key)

                    _child_region = region_map.get(_child_id)
                    _promoted_sec = section_for_region.get(_child_id) or section_for_region.get(q_node.region_id)
                    _promoted_q = Question(
                        id=_promoted_id,
                        number=_promoted_num,
                        text=_child_text,
                        page=_child_node.page,
                        bbox=_child_node.bbox,
                        order_index=order_counter,
                        section=_promoted_sec,
                        question_type="SHORT_ANSWER",
                        source_region_ids=[_child_id],
                        source_regions=[Region(page=_child_node.page, bbox=_child_node.bbox)],
                        extraction_confidence=_child_node.confidence,
                        verification_state=_child_region.verification_state if _child_region else "UNVERIFIED",
                    )
                    extracted_questions.append(_promoted_q)
                    order_counter += 1
                    print(
                        f"[IntelligentExtraction] Group-parent promotion: emitted '{_promoted_num}' "
                        f"as top-level question from parent '{q_text[:40]}'."
                    )
                continue



            # Deduplication: reject same display_number with lower confidence
            _is_uuid_fallback = len(display_num) == 6 and display_num.replace('-', '').isalnum()
            if display_num and not _is_uuid_fallback:
                prior = seen_display_numbers.get(display_num)
                if prior is not None:
                    prior_region_id, prior_conf = prior
                    if q_node.confidence <= prior_conf:
                        # New node is worse — reject it
                        audit.duplicate_rejected += 1
                        print(
                            f"[IntelligentExtraction] Fix-2: Duplicate QUESTION node '{display_num}' "
                            f"(region={q_node.region_id}, conf={q_node.confidence:.2f}) "
                            f"rejected; kept region={prior_region_id} (conf={prior_conf:.2f})"
                        )
                        continue
                    else:
                        # New node is better — retroactively reject the prior one
                        rejected_region_ids.add(prior_region_id)
                        audit.duplicate_rejected += 1
                        print(
                            f"[IntelligentExtraction] Fix-2: Replacing lower-confidence QUESTION '{display_num}' "
                            f"(prior conf={prior_conf:.2f} < new conf={q_node.confidence:.2f})"
                        )
                seen_display_numbers[display_num] = (q_node.region_id, q_node.confidence)

            # Spatial deduplication: reject if very close to an already-extracted question
            spatial_key = (q_node.page, round(q_node.bbox.y / 5.0))
            if spatial_key in seen_spatial_keys:
                audit.duplicate_rejected += 1
                print(
                    f"[IntelligentExtraction] Fix-2: Spatial duplicate QUESTION node "
                    f"(page={q_node.page}, y≈{q_node.bbox.y:.0f}) rejected."
                )
                continue
            seen_spatial_keys.add(spatial_key)

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

            # Determine initial question type — deferred to after option attachment (FIX 3).
            # Setting SHORT_ANSWER as default; MCQ is upgraded by _attach_options;
            # LONG_ANSWER is set post-attachment only if no options found and text is very long.
            q_type = "SHORT_ANSWER"

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
                page_regions_by_page=page_regions_by_page,
            )
            if option_count > 0:
                q_obj.question_type = "MCQ"
            else:
                # FIX 3: Post-classification — only upgrade to LONG_ANSWER when no options found
                # and text is substantially long (threshold raised to avoid penalising MCQ stems).
                if len(q_obj.text) > 200:
                    q_obj.question_type = "LONG_ANSWER"

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

        # FIX 5: Remove any entries that were retroactively rejected by a higher-confidence duplicate
        if rejected_region_ids:
            extracted_questions = [
                q for q in extracted_questions
                if not (q.source_region_ids and q.source_region_ids[0] in rejected_region_ids)
            ]

        # ORPHAN SUBQUESTION SWEEP
        # Walk every SUBQUESTION node in the graph.  If its parent QUESTION was never
        # emitted (e.g. because the VLM correctly labelled the parent as INSTRUCTION, or
        # because the parent was rejected by the group-parent guard without any
        # subquestion_of edges we could walk at that time), promote the orphan to a
        # top-level question so it can be matched to student answers.
        _emitted_region_ids = {
            rid for q in extracted_questions for rid in (q.source_region_ids or [])
        }
        # Build a map: child_region_id → parent_region_id for subquestion_of edges
        _subq_to_parent: Dict[str, str] = {}
        for edge in graph.edges:
            if edge.relationship == "subquestion_of":
                _subq_to_parent[edge.source_id] = edge.target_id

        for node_id, node in graph.nodes.items():
            if node.role not in ("SUBQUESTION", "QUESTION"):
                continue
            # Only handle nodes NOT yet emitted
            if node_id in _emitted_region_ids:
                continue
            # Only handle nodes whose parent is ALSO not emitted (true orphan)
            parent_id = _subq_to_parent.get(node_id)
            if parent_id and parent_id in _emitted_region_ids:
                continue  # parent was emitted; subquestion is attached as child — skip

            node_text = node.text.strip()
            if not node_text:
                continue

            # Fragment guard — skip bare markers like "(i)", "(a)" with no body
            _frag_re = re.compile(
                r'^\s*[\(\[]?\s*(?:i{1,4}v?|vi{0,3}|ix|x{1,3}|[a-dA-D]|[ivxlcdm]{1,4})\s*[\)\]\.\:]?\s*$',
                re.IGNORECASE
            )
            if _frag_re.match(node_text) and len(node_text) <= 8:
                continue

            # Derive parent number from edge target, or infer from text
            _orphan_parent_num = "1"
            if parent_id:
                _p_node = graph.nodes.get(parent_id)
                if _p_node:
                    _pm = re.match(r'^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})', _p_node.text.strip(), re.IGNORECASE)
                    if _pm:
                        _orphan_parent_num = _pm.group(1)

            # Derive subquestion label from text
            _sub_m2 = re.match(
                r'^\s*(?:(\d{1,3})\s*[\.\-]?\s*)?[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)',
                node_text, re.IGNORECASE
            )
            if _sub_m2:
                _main_n = _sub_m2.group(1) or _orphan_parent_num
                _sub_l  = _sub_m2.group(2).lower()
                _orphan_num = f"{_main_n}({_sub_l})"
            else:
                # Can't derive a structured number — use full text prefix as key
                _orphan_num = node_text[:15]

            if _orphan_num in seen_display_numbers:
                continue
            _sp_key2 = (node.page, round(node.bbox.y / 5.0))
            if _sp_key2 in seen_spatial_keys:
                continue

            seen_display_numbers[_orphan_num] = (node_id, node.confidence)
            seen_spatial_keys.add(_sp_key2)

            _orphan_region = region_map.get(node_id)
            _orphan_sec = section_for_region.get(node_id)
            _orphan_q = Question(
                id=f"Q{_orphan_num.replace(' ', '')}",
                number=_orphan_num,
                text=node_text,
                page=node.page,
                bbox=node.bbox,
                order_index=order_counter,
                section=_orphan_sec,
                question_type="SHORT_ANSWER",
                source_region_ids=[node_id],
                source_regions=[Region(page=node.page, bbox=node.bbox)],
                extraction_confidence=node.confidence,
                verification_state=_orphan_region.verification_state if _orphan_region else "UNVERIFIED",
            )
            extracted_questions.append(_orphan_q)
            order_counter += 1
            print(
                f"[IntelligentExtraction] Orphan sweep: promoted orphan subquestion '{_orphan_num}' "
                f"(region={node_id}) to top-level question."
            )


        # Record non-question rejections for audit — preserving all semantic visual structures
        _AUDIT_NON_QUESTION_ROLES = {
            "HEADER", "FOOTER", "METADATA", "INSTRUCTION", "SECTION_HEADER",
            "TABLE", "DIAGRAM", "FIGURE", "CAPTION", "FORM_FIELD",
            "PARAGRAPH", "ANSWER_REGION", "HANDWRITING", "SIGNATURE", "LIST",
        }
        for node_id, node in graph.nodes.items():
            if node.role in _AUDIT_NON_QUESTION_ROLES:
                reason_label = (
                    "administrative" if node.role in ("HEADER", "FOOTER", "METADATA", "INSTRUCTION")
                    else "section" if node.role == "SECTION_HEADER"
                    else "visual-structure"
                )
                rejection_records.append(RejectionRecord(
                    region_id=node_id,
                    ocr_text=node.text[:60],
                    classification=node.role,
                    confidence=node.confidence,
                    reason=f"{reason_label.capitalize()} {node.role.lower()} content — not a question.",
                ))
                audit.rejected_count += 1

        # Re-sort by natural question order (page, numeric_q_num, Y, X)
        def _get_q_sort_key(q: Question):
            pg = q.source_regions[0].page if q.source_regions else q.page
            y = q.bbox.y if q.bbox else 0
            x = q.bbox.x if q.bbox else 0
            m_num = re.match(r"^\s*(\d{1,3})", str(q.number or ""))
            num_val = int(m_num.group(1)) if m_num else 999
            return (pg, num_val, y, x)

        extracted_questions.sort(key=_get_q_sort_key)
        for _oi, _q in enumerate(extracted_questions):
            _q.order_index = _oi

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

    def _extract_display_number(self, text: str, active_parent_num: str = "1", fallback_index: Optional[int] = None) -> str:
        """Extracts the display question number from text like 'Q1. Explain...' -> '1' or 'a) Define...' -> '1(a)'."""
        # 0. Clean out any leading hex UUID prefix (e.g., 'bb78 ', '3fa8 ') if followed by real question text
        clean_t = re.sub(r'^[a-f0-9]{4,8}\s+(?=.*[A-Za-z0-9])', '', text.strip(), flags=re.IGNORECASE)

        # Subquestion pattern with main digit: 1(a) or Q1(a) or (1)(a)
        m_sub = re.match(
            r"^\s*(?:Q(?:uestion)?\.?\s*)?[\(\[]?(\d{1,3})[\)\]]?\s*[\.\:\-]?\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]",
            clean_t, re.IGNORECASE
        )
        if m_sub:
            return f"{m_sub.group(1)}({m_sub.group(2).lower()})"

        # Standalone subquestion letter: a) or (a) or a.
        m_let = re.match(
            r"^\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]",
            clean_t, re.IGNORECASE
        )
        if m_let:
            return f"{active_parent_num}({m_let.group(1).lower()})"

        # Primary main question number: Q1. or 1. or 1) or (1) or [1] or 1:
        m = re.match(
            r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?[\(\[]?\s*(\d{1,3})\s*[\)\]\.\:\-]",
            clean_t, re.IGNORECASE
        )
        if m:
            return m.group(1)

        # Fallback: search for first isolated integer in header text
        m_search = re.search(r"^\s*(?:Q(?:uestion)?\.?\s*)?[\(\[]?\s*(\d{1,3})\b", clean_t, re.IGNORECASE)
        if m_search:
            return m_search.group(1)

        if fallback_index is not None:
            return str(fallback_index)

        return active_parent_num

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
                # FIX 4: Skip SUBQUESTION nodes — they must not be absorbed as
                # continuation text into the parent question. OPTION nodes are
                # already excluded; belt-and-suspenders adds SUBQUESTION.
                if child_node and child_node.role not in ("QUESTION", "SECTION_HEADER", "OPTION", "SUBQUESTION"):
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
        page_regions_by_page: Optional[Dict[int, List["DocumentRegion"]]] = None,
    ) -> int:
        """
        Attaches MCQ options to a question from graph edges.

        Three-layer option text recovery (fixes 'A B C D label-only' bug):
          1. Head text — what the OPTION GraphNode carries directly.
             (May be pre-merged by document_understanding_service if multi-block.)
          2. Continuation chain — continuation_of children in the graph.
          3. Spatial scan — same-row right-adjacent blocks when still label-only.
        """
        option_count = 0
        # FIX 2: Structural-boundary guard instead of strict OPTION role check.
        # VLM-confirmed option_of edges target option nodes whose GraphNode role may be
        # UNKNOWN (demoted by COMPLETE pass or decomposition), so reject only nodes that
        # are unambiguously a different semantic entity (QUESTION, SECTION_HEADER, HEADER, FOOTER).
        _OPTION_EXCLUSION_ROLES = {"QUESTION", "SECTION_HEADER", "HEADER", "FOOTER"}
        # Track seen normalized option bodies for text-level deduplication
        seen_option_bodies: set = set()

        for child_id, rel_type, conf in children_of.get(question_node_id, []):
            if rel_type != "option_of":
                continue

            child_node = graph_nodes.get(child_id)
            if not child_node or child_node.role in _OPTION_EXCLUSION_ROLES:
                continue

            region = region_map.get(child_id)

            # FIX 2 refinement: If the node role is UNKNOWN and it is already a consumed
            # constituent of another semantic entity (parent_region_id is set by the grounding
            # pass), skip it — its text has already been merged into the parent entity.
            if child_node.role == "UNKNOWN" and region and getattr(region, "parent_region_id", None):
                continue

            # Layer 1: head text (prefer grounded region text from doc_understanding_service)
            head_text = (region.text if region and region.text.strip() else child_node.text).strip()

            # Layer 2: collect continuation children from graph
            continuation_parts: list = []
            for cont_id, cont_rel, _ in children_of.get(child_id, []):
                if cont_rel == "continuation_of":
                    cont_node = graph_nodes.get(cont_id)
                    if cont_node and cont_node.text.strip():
                        continuation_parts.append(
                            (cont_node.bbox.y if cont_node.bbox else 0, cont_node.text.strip())
                        )
            continuation_parts.sort(key=lambda t: t[0])
            cont_body = " ".join(p[1] for p in continuation_parts)

            if cont_body and cont_body not in head_text:
                opt_text = f"{head_text} {cont_body}".strip()
            else:
                opt_text = head_text

            # Layer 3: spatial fallback if still looks like bare label
            _bare_label = re.compile(r'^\s*[\(\[]?\s*[A-Da-d1-4]\s*[\)\]\.:]?\s*$')
            if _bare_label.match(opt_text) and child_node.bbox and page_regions_by_page:
                page_regs = page_regions_by_page.get(child_node.page, [])
                label_y = child_node.bbox.y
                label_x_right = child_node.bbox.x + child_node.bbox.width
                label_height = max(child_node.bbox.height, 12.0)
                # Same-row blocks that are immediately to the right within the same column
                row_candidates = sorted(
                    [
                        r for r in page_regs
                        if r.region_id != child_id
                        and abs(r.bbox.y - label_y) <= label_height * 1.2
                        and label_x_right - 5 <= r.bbox.x <= label_x_right + 300.0
                        and r.region_type not in ("QUESTION", "OPTION", "SECTION_HEADER", "HEADER", "FOOTER")
                        and r.text.strip()
                    ],
                    key=lambda r: r.bbox.x,
                )
                if row_candidates:
                    spatial_body = " ".join(r.text.strip() for r in row_candidates[:2])
                    opt_text = f"{opt_text} {spatial_body}".strip()
                    print(
                        f"[IntelligentExtraction] Spatial fallback: option '{child_id}' label-only; "
                        f"merged body '{spatial_body[:50]}'"
                    )

            # Parse label and body
            opt_m = OPTION_PREFIX_RE.match(opt_text)
            label = opt_m.group(1).upper() if opt_m else chr(65 + option_count)
            text_val = opt_m.group(2).strip() if opt_m else opt_text

            # Last resort: if text_val still empty, use cont_body
            if not text_val and cont_body:
                text_val = cont_body

            # Text-level deduplication: skip if the same option body already added
            # (can occur when decomposition crops produce the same option structure twice)
            _norm_body = text_val.lower().strip()
            if _norm_body and _norm_body in seen_option_bodies:
                continue
            if _norm_body:
                seen_option_bodies.add(_norm_body)

            opt_id = f"opt_{question.id}_{label}_{uuid.uuid4().hex[:4]}"
            full_opt_text = f"{label}. {text_val}" if text_val else opt_text
            extracted_opt = ExtractedOption(
                option_id=opt_id,
                question_id=question.id,
                label=label,
                text=full_opt_text,
                source_region_ids=[child_id],
                source_regions=[Region(page=child_node.page, bbox=child_node.bbox)],
                extraction_confidence=child_node.confidence,
                verification_state=region.verification_state if region else "UNVERIFIED",
            )
            question.extracted_options.append(extracted_opt)
            question.options.append(full_opt_text)
            option_count += 1

        return option_count

    def _attach_subquestions(
        self,
        parent_question: Question,
        question_node_id: str,
        children_of: Dict[str, List[Tuple[str, str, float]]],
        graph_nodes: Dict[str, GraphNode],
        region_map: Dict[str, DocumentRegion],
        document_id: str,
        extracted_questions: List[Question],  # kept in signature for compatibility — NOT appended to
        order_counter_ref: List[int],
        sec_title: Optional[str],
    ) -> None:
        """
        Attaches subquestions as CHILDREN of their parent question.

        FIX (57→61 inflation): Previously appended sub_q to extracted_questions,
        which caused subquestions to be counted, graded, and scored as top-level
        questions. This inflated 57 QUESTION nodes into 61 graded questions.

        Subquestions now live in parent_question.subquestions ONLY.
        They are never added to the flat extracted_questions list.
        """
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

            # Attach to parent — NOT to the flat top-level list
            if not hasattr(parent_question, 'subquestions') or parent_question.subquestions is None:
                parent_question.subquestions = []
            parent_question.subquestions.append(sub_q)
            order_counter_ref[0] += 1
            print(
                f"[IntelligentExtraction] Subquestion '{display_num}' attached to parent '{parent_question.number}' "
                f"(NOT in top-level list)."
            )

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
