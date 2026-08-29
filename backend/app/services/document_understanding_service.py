"""
Document Understanding Service — VLM-First Architecture.

Architecture:
  Page Image + OCR Evidence → VLM DOCUMENT BRAIN → Evidence Fusion → Structure Graph

The VLM is the primary document-understanding intelligence.
Deterministic analysis (regex, position, keywords) provides EVIDENCE, not AUTHORITY.
The structure graph is built BEFORE question extraction and drives it.

When VLM is disabled, deterministic analysis is the sole evidence source
and the graph is built from deterministic hypotheses alone.

Preserves 100% of original OCR geometry, BBoxes, and text without alteration.
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple, Any, Set

from app.core.config import settings
from app.models.schemas import (
    Block,
    BBox,
    DocumentRegion,
    DocumentPage,
    DocumentObservation,
    DocumentUnderstandingResult,
    StructureHypothesis,
    DocumentEvidence,
    RegionRelationship,
    DocumentRegionType,
    RelationshipType,
    CostAccounting,
    VisualVerificationResponse,
    VLMPageUnderstanding,
    VLMStructureItem,
    DocumentStructureGraph,
    GraphNode,
    GraphEdge,
)
from app.services.document_vision_provider import DocumentVisionProvider, MultimodalDocumentVisionProvider
from app.services.embedding_service import embed_texts
from app.services.evidence_fusion_service import EvidenceFusionService


class DocumentUnderstandingService:
    """
    Orchestration service for building structured document representations.

    Flow:
    1. Create DocumentRegions from OCR blocks (preserving exact geometry)
    2. Generate deterministic hypotheses (as EVIDENCE, not authority)
    3. If VLM enabled: call understand_page() for EACH page
    4. Fuse VLM understanding + deterministic evidence
    5. Build DocumentStructureGraph from fused evidence
    6. Return result with populated graph for downstream extraction
    """

    def __init__(self, vision_provider: Optional[DocumentVisionProvider] = None):
        self.vision_provider = vision_provider or MultimodalDocumentVisionProvider()
        self.fusion_service = EvidenceFusionService()

    def process_document(
        self,
        blocks: List[Block],
        document_id: str = "doc_1",
        page_sizes: Optional[Dict[int, List[float]]] = None,
        page_images: Optional[Dict[int, Any]] = None,
        attach_embeddings: bool = True,
        force_vlm_verification: bool = False,
    ) -> DocumentUnderstandingResult:
        """
        Converts low-level OCR blocks into a verified DocumentUnderstandingResult
        with a populated DocumentStructureGraph.
        """
        if not blocks:
            return DocumentUnderstandingResult(
                document_id=document_id,
                pages=[],
                regions=[],
                relationships=[],
                conflicts=[],
                vlm_status="NOT_CONFIGURED",
                cost_accounting=CostAccounting(),
                metadata={"block_count": 0},
            )

        # 1. Group blocks by page
        pages_dict: Dict[int, List[Block]] = {}
        for block in blocks:
            pages_dict.setdefault(block.page, []).append(block)

        all_doc_regions: List[DocumentRegion] = []
        doc_pages: List[DocumentPage] = []
        all_relationships: List[RegionRelationship] = []
        global_conflicts: List[Dict[str, Any]] = []

        # 2. Create regions from blocks and generate deterministic evidence
        for page_num in sorted(pages_dict.keys()):
            page_blocks = pages_dict[page_num]
            p_width = 1000.0
            p_height = 1000.0
            if page_sizes and page_num in page_sizes:
                p_width, p_height = page_sizes[page_num][0], page_sizes[page_num][1]

            page_regions, page_rels, page_conflicts = self._process_page(
                page_blocks=page_blocks,
                page_num=page_num,
                page_width=p_width,
                page_height=p_height,
            )

            all_doc_regions.extend(page_regions)
            all_relationships.extend(page_rels)
            global_conflicts.extend(page_conflicts)

            reading_order = [r.region_id for r in page_regions]
            doc_pages.append(DocumentPage(
                page_number=page_num,
                width=p_width,
                height=p_height,
                regions=page_regions,
                reading_order=reading_order,
            ))

        # 3. Cross-page relationships
        cross_page_rels = self._extract_cross_page_relationships(doc_pages)
        all_relationships.extend(cross_page_rels)

        # 4. Attach embeddings
        if attach_embeddings and getattr(settings, "EMBEDDING_ENGINE_ENABLED", True):
            self._attach_embeddings(all_doc_regions)

        # 5. PAGE-LEVEL VLM DOCUMENT UNDERSTANDING
        vlm_page_understandings: List[VLMPageUnderstanding] = []
        vlm_enabled = getattr(settings, "DOCUMENT_VLM_ENABLED", False) or force_vlm_verification
        vlm_page_mode = getattr(settings, "DOCUMENT_VLM_PAGE_UNDERSTANDING", True)
        total_pages = len(doc_pages)
        cost = CostAccounting(
            pages_considered=total_pages,
            regions_considered=len(all_doc_regions),
        )

        if vlm_enabled and vlm_page_mode and isinstance(self.vision_provider, MultimodalDocumentVisionProvider):
            print(f"[DocUnderstanding] VLM Page Intelligence: Processing {total_pages} page(s)")

            for page_obj in doc_pages:
                page_num = page_obj.page_number
                page_blocks = pages_dict.get(page_num, [])

                # Get page image
                page_img = None
                if page_images and page_num in page_images:
                    page_img = page_images[page_num]

                # Build page context from neighboring pages
                page_context = self._build_page_context(page_num, pages_dict, total_pages)

                # VLM understands this page
                understanding = self.vision_provider.understand_page(
                    page_image=page_img,
                    ocr_blocks=page_blocks,
                    page_number=page_num,
                    total_pages=total_pages,
                    page_context=page_context,
                    force_vlm=force_vlm_verification,
                )

                vlm_page_understandings.append(understanding)
                cost.vlm_calls += 1
                cost.pages_sent += 1 if understanding.image_sent else 0
                cost.regions_sent += understanding.ocr_blocks_sent

                if understanding.structures:
                    cost.successful_calls += 1
                    print(f"[DocUnderstanding] Page {page_num}: VLM identified {len(understanding.structures)} structures, {len(understanding.relationships)} relationships")
                else:
                    print(f"[DocUnderstanding] Page {page_num}: VLM returned no structures")

            # 6. APPLY VLM UNDERSTANDING — VLM is the primary intelligence
            self._apply_vlm_page_understandings(
                vlm_understandings=vlm_page_understandings,
                all_regions=all_doc_regions,
                all_relationships=all_relationships,
                pages_dict=pages_dict,
            )

            vlm_status = "SUCCESS" if any(u.structures for u in vlm_page_understandings) else "VLM_NO_STRUCTURES"
        elif vlm_enabled and not vlm_page_mode:
            # Legacy selective verification mode
            vlm_status = "LEGACY_MODE"
        else:
            vlm_status = "NOT_CONFIGURED"

        # 7. BUILD STRUCTURE GRAPH — this drives extraction
        raw_inferred_purpose = "UNKNOWN"
        inferred_roles: Dict[int, str] = {}
        for u in vlm_page_understandings:
            if u.document_purpose and u.document_purpose != "UNKNOWN":
                raw_inferred_purpose = u.document_purpose
            if u.page_purpose and u.page_purpose != "UNKNOWN":
                inferred_roles[u.page_number] = u.page_purpose

        # Normalize purpose to schema DocumentPurpose literal
        purpose_map = {
            "EXAMINATION_PAPER": "QUESTION_PAPER",
            "QUESTION_PAPER": "QUESTION_PAPER",
            "EXAM": "QUESTION_PAPER",
            "ASSIGNMENT": "QUESTION_PAPER",
            "INSTRUCTIONS": "INSTRUCTIONS",
            "COVER": "COVER",
            "ANSWER_KEY": "ANSWER_KEY",
            "REFERENCE": "REFERENCE",
        }
        inferred_purpose = purpose_map.get(raw_inferred_purpose, "UNKNOWN")

        # Fallback: if no VLM, infer from deterministic evidence
        if inferred_purpose == "UNKNOWN":
            q_count = sum(1 for r in all_doc_regions if r.region_type == "QUESTION")
            if q_count > 0:
                inferred_purpose = "QUESTION_PAPER"

        for page_obj in doc_pages:
            if page_obj.page_number not in inferred_roles:
                page_regs = [r for r in all_doc_regions if r.page == page_obj.page_number]
                q_on_page = sum(1 for r in page_regs if r.region_type == "QUESTION")
                if q_on_page > 0:
                    inferred_roles[page_obj.page_number] = "QUESTION_PAGE"
                else:
                    meta_on_page = sum(1 for r in page_regs if r.region_type in ("HEADER", "METADATA", "INSTRUCTION", "FOOTER"))
                    if meta_on_page > len(page_regs) * 0.6:
                        inferred_roles[page_obj.page_number] = "ADMINISTRATIVE"
                    else:
                        inferred_roles[page_obj.page_number] = "MIXED"

        structure_graph = self._build_structure_graph(
            all_regions=all_doc_regions,
            all_relationships=all_relationships,
            document_purpose=inferred_purpose,
            page_roles=inferred_roles,
        )

        # 8. Evidence Fusion
        result = DocumentUnderstandingResult(
            document_id=document_id,
            pages=doc_pages,
            regions=all_doc_regions,
            relationships=all_relationships,
            conflicts=global_conflicts,
            vlm_status=vlm_status,
            cost_accounting=cost,
            structure_graph=structure_graph,
            vlm_page_understandings=vlm_page_understandings,
            document_purpose=inferred_purpose,
            page_roles=inferred_roles,
            metadata={
                "total_blocks": len(blocks),
                "total_pages": total_pages,
                "total_regions": len(all_doc_regions),
                "total_relationships": len(all_relationships),
                "total_conflicts": len(global_conflicts),
                "vlm_pages_processed": len(vlm_page_understandings),
                "vlm_structures_found": sum(len(u.structures) for u in vlm_page_understandings),
            },
        )

        # Apply evidence fusion (upgrades verification states)
        vlm_response = None  # Page understanding replaces old verify_structure response
        final_result = self.fusion_service.fuse_document_evidence(result, vlm_response)

        return final_result

    def _build_page_context(
        self, page_num: int, pages_dict: Dict[int, List[Block]], total_pages: int
    ) -> Dict[str, Any]:
        """Builds context from neighboring pages for VLM."""
        context: Dict[str, Any] = {}

        # Previous page summary
        if page_num > 1 and (page_num - 1) in pages_dict:
            prev_blocks = pages_dict[page_num - 1]
            prev_texts = [b.text for b in sorted(prev_blocks, key=lambda b: b.bbox.y)[:5]]
            context["prev_page_summary"] = " | ".join(prev_texts)[:200]

        # Next page summary
        if page_num < total_pages and (page_num + 1) in pages_dict:
            next_blocks = pages_dict[page_num + 1]
            next_texts = [b.text for b in sorted(next_blocks, key=lambda b: b.bbox.y)[:5]]
            context["next_page_summary"] = " | ".join(next_texts)[:200]

        return context

    def _apply_vlm_page_understandings(
        self,
        vlm_understandings: List[VLMPageUnderstanding],
        all_regions: List[DocumentRegion],
        all_relationships: List[RegionRelationship],
        pages_dict: Dict[int, List[Block]],
    ) -> None:
        """
        Applies VLM page understanding as the PRIMARY intelligence source.

        The VLM's structural analysis OVERRIDES deterministic hypotheses
        when the VLM has sufficient confidence.

        Deterministic analysis remains as a fallback evidence source
        for regions the VLM didn't address.
        """
        region_map = {r.region_id: r for r in all_regions}

        for understanding in vlm_understandings:
            if not understanding.structures:
                continue

            # Apply VLM structural decisions to regions
            vlm_assigned_ids: Set[str] = set()

            for struct in understanding.structures:
                if not struct.region_ids:
                    continue

                head_id = struct.region_ids[0]
                cont_ids = struct.region_ids[1:]

                # Process head region
                if head_id in region_map:
                    reg = region_map[head_id]
                    vlm_assigned_ids.add(head_id)

                    vlm_hyp = StructureHypothesis(
                        region_id=head_id,
                        hypothesized_type=struct.role,
                        confidence=struct.confidence,
                        source="vlm_page_understanding",
                        evidence=[DocumentEvidence(
                            signal_type="visual_vlm_verification",
                            description=f"VLM page understanding: {struct.reasoning}",
                            weight=0.9,
                            score=struct.confidence,
                        )],
                    )

                    existing_sources = {h.source for h in reg.conflicting_hypotheses}
                    if "vlm_page_understanding" not in existing_sources:
                        reg.conflicting_hypotheses.append(vlm_hyp)
                    reg.vlm_hypothesis = vlm_hyp

                    if struct.confidence >= 0.60:
                        det_agrees = any(
                            h.hypothesized_type == struct.role
                            for h in reg.conflicting_hypotheses
                            if h.source != "vlm_page_understanding"
                        )
                        reg.region_type = struct.role
                        reg.confidence = min(1.0, struct.confidence + 0.05) if det_agrees else struct.confidence
                        reg.verification_state = "VERIFIED"
                        reg.classification_conflict = False
                    else:
                        reg.verification_state = "UNCERTAIN"
                        reg.uncertainty = 1.0 - struct.confidence

                    reg.evidence.append(DocumentEvidence(
                        signal_type="visual_vlm_verification",
                        description=f"VLM: {struct.reasoning[:100]}",
                        weight=0.9,
                        score=struct.confidence,
                    ))

                # Process trailing continuation regions in multi-region structure
                for cont_id in cont_ids:
                    if cont_id not in region_map:
                        continue
                    vlm_assigned_ids.add(cont_id)
                    cont_reg = region_map[cont_id]
                    # Continuation regions stay non-QUESTION so they don't spawn duplicate questions
                    cont_reg.region_type = "UNKNOWN"
                    cont_reg.verification_state = "VERIFIED"
                    cont_reg.parent_region_id = head_id
                    # Add continuation edge
                    all_relationships.append(RegionRelationship(
                        source_region_id=cont_id,
                        target_region_id=head_id,
                        relationship_type="continuation_of",
                        confidence=struct.confidence,
                        evidence=[DocumentEvidence(
                            signal_type="visual_vlm_verification",
                            description=f"Multi-region structure continuation of {head_id}",
                            weight=0.9,
                            score=struct.confidence,
                        )],
                    ))

            # Apply VLM relationships
            for rel in understanding.relationships:
                for src_id in rel.source_ids:
                    for tgt_id in rel.target_ids:
                        if src_id in region_map and tgt_id in region_map:
                            # Check for duplicates
                            existing_key = (src_id, tgt_id, rel.relationship_type)
                            already_exists = any(
                                (r.source_region_id, r.target_region_id, r.relationship_type) == existing_key
                                for r in all_relationships
                            )
                            if not already_exists:
                                all_relationships.append(RegionRelationship(
                                    source_region_id=src_id,
                                    target_region_id=tgt_id,
                                    relationship_type=rel.relationship_type,
                                    confidence=rel.confidence,
                                    evidence=[DocumentEvidence(
                                        signal_type="visual_vlm_verification",
                                        description=f"VLM relationship: {rel.relationship_type}",
                                        weight=0.9,
                                        score=rel.confidence,
                                    )],
                                ))

    def _build_structure_graph(
        self,
        all_regions: List[DocumentRegion],
        all_relationships: List[RegionRelationship],
        document_purpose: str,
        page_roles: Dict[int, str],
    ) -> DocumentStructureGraph:
        """
        Constructs the DocumentStructureGraph from fused evidence.

        This graph is the PRIMARY input for question extraction.
        It is built BEFORE extraction, not as an output report.
        """
        nodes: Dict[str, GraphNode] = {}
        for r in all_regions:
            nodes[r.region_id] = GraphNode(
                region_id=r.region_id,
                role=r.region_type,
                text=r.text,
                page=r.page,
                bbox=r.bbox,
                confidence=r.confidence,
            )

        edges: List[GraphEdge] = []
        for rel in all_relationships:
            evidence_sources = [e.signal_type for e in rel.evidence] if rel.evidence else []
            edges.append(GraphEdge(
                source_id=rel.source_region_id,
                target_id=rel.target_region_id,
                relationship=rel.relationship_type,
                confidence=rel.confidence,
                evidence_sources=evidence_sources,
            ))

        return DocumentStructureGraph(
            nodes=nodes,
            edges=edges,
            document_purpose=document_purpose,
            page_roles=page_roles,
        )

    # Legacy method preserved for backward compatibility
    def build_structure_graph(self, doc_result: DocumentUnderstandingResult) -> DocumentStructureGraph:
        """Legacy method: builds graph from a DocumentUnderstandingResult."""
        return self._build_structure_graph(
            all_regions=doc_result.regions,
            all_relationships=doc_result.relationships,
            document_purpose=doc_result.document_purpose or "UNKNOWN",
            page_roles=doc_result.page_roles or {},
        )

    def _process_page(
        self,
        page_blocks: List[Block],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> Tuple[List[DocumentRegion], List[RegionRelationship], List[Dict[str, Any]]]:
        """Processes blocks on a single page into DocumentRegions with deterministic evidence."""
        doc_regions: List[DocumentRegion] = []
        page_relationships: List[RegionRelationship] = []
        page_conflicts: List[Dict[str, Any]] = []

        # Create DocumentRegion for each Block preserving exact BBox & text
        for b in page_blocks:
            exact_bbox = BBox(x=b.bbox.x, y=b.bbox.y, width=b.bbox.width, height=b.bbox.height)
            reg = DocumentRegion(
                region_id=b.id,
                page=b.page,
                text=b.text,
                bbox=exact_bbox,
                region_type="UNKNOWN",
                source=b.source or "ocr",
                confidence=b.confidence,
                evidence=[],
                relationships=[],
                uncertainty=0.0,
                classification_conflict=False,
                conflicting_hypotheses=[],
                verification_state="UNVERIFIED",
                metadata={"modality": b.modality, "role": b.role},
            )
            doc_regions.append(reg)

        # Generate deterministic hypotheses (EVIDENCE, not authority)
        for reg in doc_regions:
            self._analyze_region_hypotheses(reg, page_width, page_height)
            if reg.classification_conflict:
                page_conflicts.append({
                    "region_id": reg.region_id,
                    "page": reg.page,
                    "text": reg.text[:60],
                    "hypotheses": [
                        {"type": h.hypothesized_type, "confidence": h.confidence, "source": h.source}
                        for h in reg.conflicting_hypotheses
                    ],
                })

        # Establish intra-page relationships
        intra_rels = self._extract_intra_page_relationships(doc_regions)
        page_relationships.extend(intra_rels)

        rel_map: Dict[str, List[RegionRelationship]] = {}
        for rel in intra_rels:
            rel_map.setdefault(rel.source_region_id, []).append(rel)
        for reg in doc_regions:
            reg.relationships = rel_map.get(reg.region_id, [])

        return doc_regions, page_relationships, page_conflicts

    def _analyze_region_hypotheses(
        self, reg: DocumentRegion, page_width: float, page_height: float
    ) -> None:
        """
        Generates deterministic hypotheses for a region.
        These are EVIDENCE sources, not final classifications.
        The VLM's page understanding takes priority when available.
        """
        text = reg.text.strip()
        bbox = reg.bbox
        hypotheses: List[StructureHypothesis] = []

        # 1. PARSER HYPOTHESIS (Text-pattern based)
        parser_ev: List[DocumentEvidence] = []
        parser_type: DocumentRegionType = "UNKNOWN"
        parser_conf = 0.5

        q_num_match = re.search(
            r"^(?:Q(?:uestion)?[\s\.\\:]*)?\d+[\.\)\:\s]+", text, re.IGNORECASE
        )
        subq_match = re.search(r"^\(?([a-z]|[ivxlcdm]+)\)[\.\:\s]+", text, re.IGNORECASE)
        interrogative_match = re.search(
            r"\b(what|why|how|explain|describe|calculate|evaluate|find|prove|derive|compare|define|list|state|discuss|show|write|determine|solve|select|choose|identify|mark|indicate)\b",
            text, re.IGNORECASE,
        )

        if q_num_match:
            parser_ev.append(DocumentEvidence(
                signal_type="numbering_pattern",
                description=f"Matched primary question numbering pattern '{q_num_match.group(0).strip()}'",
                weight=0.4, score=0.9,
            ))
        if subq_match and not q_num_match:
            parser_ev.append(DocumentEvidence(
                signal_type="numbering_pattern",
                description=f"Matched subquestion numbering pattern '{subq_match.group(0).strip()}'",
                weight=0.4, score=0.85,
            ))
        if interrogative_match:
            parser_ev.append(DocumentEvidence(
                signal_type="question_interrogative",
                description=f"Contains question keyword '{interrogative_match.group(1)}'",
                weight=0.3, score=0.8,
            ))

        opt_match = re.search(
            r"^\(?[A-Da-d1-9i-zIVXLCDM]+\)[\.\:\s]+|^(?:Option|Choice)\s+[\(]?[A-Za-z0-9ivxlcdm]+\)?[\.\:\s]*",
            text, re.IGNORECASE,
        )
        if opt_match and not interrogative_match:
            parser_ev.append(DocumentEvidence(
                signal_type="option_formatting",
                description=f"Matched option pattern '{opt_match.group(0).strip()}'",
                weight=0.5, score=0.95,
            ))

        inst_match = re.search(
            r"\b(?:Note|Instructions|General Instructions|Notice|Read carefully|Answer all|All questions carry|COMPULSORY|carrying \w+ marks|attempt any \w+ questions)\b",
            text, re.IGNORECASE,
        )
        if inst_match:
            parser_ev.append(DocumentEvidence(
                signal_type="section_formatting",
                description=f"Matched instruction keyword pattern",
                weight=0.5, score=0.9,
            ))

        sec_match = re.search(
            r"\b(?:SECTION|PART|GROUP|UNIT|CHAPTER)\s*[\-–\s]*[A-Z0-9IVX]+", text, re.IGNORECASE
        )
        if sec_match:
            parser_ev.append(DocumentEvidence(
                signal_type="heading_formatting",
                description=f"Matched section header pattern '{sec_match.group(0).strip()}'",
                weight=0.5, score=0.95,
            ))

        if sec_match and not interrogative_match:
            parser_type = "SECTION_HEADER"; parser_conf = 0.95
        elif inst_match and not interrogative_match:
            parser_type = "INSTRUCTION"; parser_conf = 0.90
        elif opt_match and not interrogative_match:
            parser_type = "OPTION"; parser_conf = 0.88
        elif q_num_match and interrogative_match:
            parser_type = "QUESTION"; parser_conf = 0.92
        elif q_num_match:
            parser_type = "QUESTION"; parser_conf = 0.80
        elif subq_match:
            parser_type = "SUBQUESTION"; parser_conf = 0.82
        elif interrogative_match:
            parser_type = "QUESTION"; parser_conf = 0.70

        if parser_type != "UNKNOWN":
            hypotheses.append(StructureHypothesis(
                region_id=reg.region_id,
                hypothesized_type=parser_type,
                confidence=parser_conf,
                source="parser",
                evidence=parser_ev,
            ))

        # 2. LAYOUT ANALYZER HYPOTHESIS
        layout_ev: List[DocumentEvidence] = []
        layout_type: DocumentRegionType = "UNKNOWN"
        layout_conf = 0.50

        rel_top = bbox.y / page_height if page_height > 0 else 0
        rel_bottom = (bbox.y + bbox.height) / page_height if page_height > 0 else 0

        if rel_top < 0.06:
            layout_ev.append(DocumentEvidence(
                signal_type="page_position",
                description=f"Positioned near top margin (rel_y={rel_top:.3f})",
                weight=0.4, score=0.85,
            ))
            layout_type = "HEADER"; layout_conf = 0.85
        elif rel_bottom > 0.94:
            layout_ev.append(DocumentEvidence(
                signal_type="page_position",
                description=f"Positioned near bottom margin (rel_y={rel_bottom:.3f})",
                weight=0.4, score=0.85,
            ))
            layout_type = "FOOTER"; layout_conf = 0.85

        if reg.metadata.get("role") == "visual_element" or re.search(
            r"\[(?:Diagram|Figure|Image|Graph)\]|\b(?:Fig\.|Figure|Diagram)\s*\d+", text, re.IGNORECASE
        ):
            layout_ev.append(DocumentEvidence(
                signal_type="table_geometry",
                description="Visual element / Diagram reference detected",
                weight=0.5, score=0.9,
            ))
            layout_type = "DIAGRAM"; layout_conf = 0.90
        elif re.search(r"^\|.*?\|", text, re.MULTILINE) or re.search(r"\bTable\s*\d*[\:\.\s]", text, re.IGNORECASE) or (
            "\t" in text and text.count("\t") >= 2
        ):
            layout_ev.append(DocumentEvidence(
                signal_type="table_geometry",
                description="Tabular syntax / grid structure detected",
                weight=0.5, score=0.88,
            ))
            layout_type = "TABLE"; layout_conf = 0.88

        if layout_type != "UNKNOWN":
            hypotheses.append(StructureHypothesis(
                region_id=reg.region_id,
                hypothesized_type=layout_type,
                confidence=layout_conf,
                source="layout_analyzer",
                evidence=layout_ev,
            ))

        # 3. SEMANTIC ANALYZER HYPOTHESIS
        if re.search(r"\b(?:Roll No|Name|Date|Subject|Time Allowed|Maximum Marks|Class|Session)\b", text, re.IGNORECASE):
            sem_ev = [DocumentEvidence(
                signal_type="semantic_signal",
                description="Document metadata header terms detected",
                weight=0.5, score=0.9,
            )]
            hypotheses.append(StructureHypothesis(
                region_id=reg.region_id,
                hypothesized_type="METADATA",
                confidence=0.90,
                source="semantic_analyzer",
                evidence=sem_ev,
            ))

        # 4. RESOLVE HYPOTHESES (deterministic only — VLM will override later)
        if not hypotheses:
            reg.region_type = "UNKNOWN"
            reg.confidence = 0.5
            reg.evidence = []
            reg.classification_conflict = False
            reg.conflicting_hypotheses = []
            return

        distinct_types = {h.hypothesized_type for h in hypotheses}
        top_hypothesis = sorted(hypotheses, key=lambda x: x.confidence, reverse=True)[0]

        if len(distinct_types) > 1:
            reg.classification_conflict = True
            reg.conflicting_hypotheses = hypotheses
            reg.uncertainty = 0.4
            reg.region_type = top_hypothesis.hypothesized_type
            reg.confidence = top_hypothesis.confidence * 0.85
            reg.evidence = [ev for h in hypotheses for ev in h.evidence]
        else:
            reg.classification_conflict = False
            reg.conflicting_hypotheses = hypotheses
            reg.uncertainty = max(0.0, 1.0 - top_hypothesis.confidence)
            reg.region_type = top_hypothesis.hypothesized_type
            reg.confidence = top_hypothesis.confidence
            reg.evidence = top_hypothesis.evidence

    def _extract_intra_page_relationships(
        self, regions: List[DocumentRegion]
    ) -> List[RegionRelationship]:
        """Extracts spatial, hierarchical, and reading order relationships on a page."""
        rels: List[RegionRelationship] = []
        if not regions:
            return rels

        sorted_regs = sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x))
        current_section_id: Optional[str] = None
        current_question_id: Optional[str] = None

        for idx, reg in enumerate(sorted_regs):
            if idx > 0:
                prev_reg = sorted_regs[idx - 1]
                rels.append(RegionRelationship(
                    source_region_id=reg.region_id,
                    target_region_id=prev_reg.region_id,
                    relationship_type="follows",
                    confidence=0.95,
                    evidence=[DocumentEvidence(
                        signal_type="spatial_position",
                        description=f"Sequentially follows region '{prev_reg.region_id}' in reading order",
                        weight=0.3, score=0.95,
                    )],
                ))

            if reg.region_type == "SECTION_HEADER" or re.search(r"^\s*(?:SECTION|PART|GROUP)\s+[A-Z0-9]", reg.text, re.IGNORECASE):
                current_section_id = reg.region_id
            elif current_section_id and reg.region_id != current_section_id:
                rels.append(RegionRelationship(
                    source_region_id=current_section_id,
                    target_region_id=reg.region_id,
                    relationship_type="section_member",
                    confidence=0.95,
                ))

            if reg.region_type == "QUESTION":
                if current_question_id and not re.search(r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?\d{1,3}\s*[\.\):\-]", reg.text, re.IGNORECASE) and not re.search(r"^\s*\(\s*[a-z0-9]+\s*\)", reg.text, re.IGNORECASE):
                    rels.append(RegionRelationship(
                        source_region_id=reg.region_id,
                        target_region_id=current_question_id,
                        relationship_type="continuation_of",
                        confidence=0.85,
                    ))
                else:
                    current_question_id = reg.region_id
            elif reg.region_type in ("UNKNOWN", "INSTRUCTION") and current_question_id and not re.search(r"^\s*(?:SECTION|PART|GROUP|Table|Figure)\b", reg.text, re.IGNORECASE):
                rels.append(RegionRelationship(
                    source_region_id=reg.region_id,
                    target_region_id=current_question_id,
                    relationship_type="continuation_of",
                    confidence=0.80,
                ))
            elif reg.region_type in ("OPTION", "SUBQUESTION", "TABLE", "DIAGRAM", "TABLE_CELL"):
                rel_t = "option_of" if reg.region_type == "OPTION" else ("subquestion_of" if reg.region_type == "SUBQUESTION" else "belongs_to")
                if current_question_id:
                    rels.append(RegionRelationship(
                        source_region_id=reg.region_id,
                        target_region_id=current_question_id,
                        relationship_type=rel_t,
                        confidence=0.90,
                    ))
                    reg.parent_region_id = current_question_id

        return rels

    def _extract_cross_page_relationships(
        self, pages: List[DocumentPage]
    ) -> List[RegionRelationship]:
        """Extracts relationships across pages (continuation_of across pages)."""
        cross_rels: List[RegionRelationship] = []
        if len(pages) < 2:
            return cross_rels

        for i in range(len(pages) - 1):
            curr_page = pages[i]
            next_page = pages[i + 1]
            if not curr_page.regions or not next_page.regions:
                continue

            last_q = next((r for r in reversed(curr_page.regions) if r.region_type == "QUESTION"), curr_page.regions[-1])
            first_reg = next_page.regions[0]

            last_text = last_q.text.strip()
            first_text = first_reg.text.strip()

            if last_text and (not last_text[-1] in (".", "?", "!", ":", ";") or not re.search(r"^\s*(?:SECTION|PART|Q\d+|\d+\.)", first_text, re.IGNORECASE)):
                cross_rels.append(RegionRelationship(
                    source_region_id=first_reg.region_id,
                    target_region_id=last_q.region_id,
                    relationship_type="continuation_of",
                    confidence=0.85,
                    evidence=[DocumentEvidence(
                        signal_type="continuation_relationship",
                        description=f"Cross-page continuation: page {next_page.page_number} → page {curr_page.page_number}",
                        weight=0.4, score=0.85,
                    )],
                ))

        return cross_rels

    def _attach_embeddings(self, regions: List[DocumentRegion]) -> None:
        """Attaches dense embeddings to regions using Step 8 embedding service."""
        texts = [r.text for r in regions]
        if not texts:
            return
        try:
            vecs = embed_texts(texts)
            if vecs is not None and len(vecs) == len(regions):
                for idx, reg in enumerate(regions):
                    reg.embedding = vecs[idx].tolist()
                    reg.evidence.append(DocumentEvidence(
                        signal_type="semantic_signal",
                        description="Dense vector embedding attached from Step 8 embedding engine",
                        weight=0.2, score=1.0,
                    ))
        except Exception as e:
            print(f"[DocumentUnderstandingService] Embedding attachment notice: {e}")


def get_debug_summary(result: DocumentUnderstandingResult) -> Dict[str, Any]:
    """Developer / debug representation helper."""
    region_summaries = []
    for r in result.regions:
        region_summaries.append({
            "region_id": r.region_id,
            "page": r.page,
            "text_snippet": r.text[:80],
            "bbox": {"x": r.bbox.x, "y": r.bbox.y, "w": r.bbox.width, "h": r.bbox.height},
            "final_region_type": r.region_type,
            "verification_state": r.verification_state,
            "confidence": round(r.confidence, 4),
            "classification_conflict": r.classification_conflict,
            "hypotheses": [
                {"source": h.source, "type": h.hypothesized_type, "confidence": round(h.confidence, 4)}
                for h in r.conflicting_hypotheses
            ],
            "evidence_count": len(r.evidence),
            "parent_region_id": r.parent_region_id,
            "child_region_ids": r.child_region_ids,
        })

    return {
        "document_id": result.document_id,
        "vlm_status": result.vlm_status,
        "document_purpose": result.document_purpose,
        "page_roles": result.page_roles,
        "total_regions": len(result.regions),
        "total_relationships": len(result.relationships),
        "conflicts_count": len(result.conflicts),
        "verification_summary": result.verification_summary,
        "cost_accounting": result.cost_accounting.model_dump() if result.cost_accounting else None,
        "regions": region_summaries,
    }
