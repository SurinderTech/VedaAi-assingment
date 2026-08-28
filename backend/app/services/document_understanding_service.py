"""
Step 11B — Universal Document Understanding Orchestration Service.

Converts OCR outputs and document page structures into a structured, evidence-backed
DocumentUnderstandingResult with rich multi-hypothesis region types, parent-child
relationships, spatial/semantic signals, conflict representation, selective VLM provider verification,
and provider-agnostic evidence fusion.

Strictly provider-agnostic: document_understanding_service.py contains zero vendor-specific code.
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
)
from app.services.document_vision_provider import DocumentVisionProvider, MultimodalDocumentVisionProvider
from app.services.embedding_service import embed_texts
from app.services.evidence_fusion_service import EvidenceFusionService


class DocumentUnderstandingService:
    """
    Orchestration service for building structured multi-modal document representations.
    Contains selective VLM routing and provider-agnostic evidence fusion.
    """

    def __init__(self, vision_provider: Optional[DocumentVisionProvider] = None):
        self.vision_provider = vision_provider or MultimodalDocumentVisionProvider()
        self.fusion_service = EvidenceFusionService()

    def process_document(
        self,
        blocks: List[Block],
        document_id: str = "doc_1",
        page_sizes: Optional[Dict[int, List[float]]] = None,
        page_images: Optional[Dict[int, bytes]] = None,
        attach_embeddings: bool = True,
        force_vlm_verification: bool = False,
    ) -> DocumentUnderstandingResult:
        """
        Converts low-level OCR blocks into a verified DocumentUnderstandingResult.
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

        # 1. Group blocks by page and preserve order
        pages_dict: Dict[int, List[Block]] = {}
        for block in blocks:
            pages_dict.setdefault(block.page, []).append(block)

        all_doc_regions: List[DocumentRegion] = []
        doc_pages: List[DocumentPage] = []
        all_relationships: List[RegionRelationship] = []
        global_conflicts: List[Dict[str, Any]] = []

        # Process each page (Step 11A initial analysis)
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
            doc_pages.append(
                DocumentPage(
                    page_number=page_num,
                    width=p_width,
                    height=p_height,
                    regions=page_regions,
                    reading_order=reading_order,
                )
            )

        # 2. Extract cross-page relationships (e.g. continuation_of across pages)
        cross_page_rels = self._extract_cross_page_relationships(doc_pages)
        all_relationships.extend(cross_page_rels)

        # 3. Attach dense vector embeddings via Step 8 embedding service if enabled
        if attach_embeddings and getattr(settings, "EMBEDDING_ENGINE_ENABLED", True):
            self._attach_embeddings(all_doc_regions)

        # 4. SELECTIVE VLM VERIFICATION ROUTER
        # Selective VLM verification logic:
        # High confidence consistent regions -> SKIP VLM
        # Ambiguous / conflicting regions -> VLM VERIFY
        target_ids_to_verify: List[str] = []
        skipped_count = 0

        for r in all_doc_regions:
            conf_threshold = getattr(settings, "DOCUMENT_VLM_CONFIDENCE_THRESHOLD", 0.80)
            is_ambiguous = r.classification_conflict or (r.confidence < conf_threshold) or (r.region_type in ("TABLE", "DIAGRAM", "UNKNOWN"))
            if force_vlm_verification or is_ambiguous:
                target_ids_to_verify.append(r.region_id)
            else:
                skipped_count += 1

        res_initial = DocumentUnderstandingResult(
            document_id=document_id,
            pages=doc_pages,
            regions=all_doc_regions,
            relationships=all_relationships,
            conflicts=global_conflicts,
            vlm_status="NOT_CONFIGURED",
            cost_accounting=CostAccounting(
                pages_considered=len(doc_pages),
                regions_considered=len(all_doc_regions),
                skipped_high_confidence_count=skipped_count,
            ),
            metadata={
                "total_blocks": len(blocks),
                "total_pages": len(doc_pages),
                "total_regions": len(all_doc_regions),
                "total_relationships": len(all_relationships),
                "total_conflicts": len(global_conflicts),
                "skipped_high_confidence": skipped_count,
                "regions_targeted_for_vlm": len(target_ids_to_verify),
            },
        )

        # Invoke VLM provider boundary only if targeted regions exist or forced
        vlm_response: Optional[VisualVerificationResponse] = None
        if target_ids_to_verify or force_vlm_verification:
            vlm_response = self.vision_provider.verify_structure(
                result=res_initial,
                page_images=page_images,
                target_region_ids=target_ids_to_verify,
            )
            res_initial.vlm_status = vlm_response.status
            if vlm_response.cost_accounting:
                res_initial.cost_accounting = vlm_response.cost_accounting

        # 5. PROVIDER-AGNOSTIC EVIDENCE FUSION
        final_result = self.fusion_service.fuse_document_evidence(res_initial, vlm_response)

        return final_result

    def _process_page(
        self,
        page_blocks: List[Block],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> Tuple[List[DocumentRegion], List[RegionRelationship], List[Dict[str, Any]]]:
        """Processes blocks on a single page into DocumentRegions and Relationships."""
        doc_regions: List[DocumentRegion] = []
        page_relationships: List[RegionRelationship] = []
        page_conflicts: List[Dict[str, Any]] = []

        # Step A: Create basic DocumentRegion for each Block preserving exact BBox & text
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

        # Step B: Gather evidence and hypotheses for each region
        for reg in doc_regions:
            self._analyze_region_hypotheses(reg, page_width, page_height)

            if reg.classification_conflict:
                page_conflicts.append(
                    {
                        "region_id": reg.region_id,
                        "page": reg.page,
                        "text": reg.text[:60],
                        "hypotheses": [
                            {"type": h.hypothesized_type, "confidence": h.confidence, "source": h.source}
                            for h in reg.conflicting_hypotheses
                        ],
                    }
                )

        # Step C: Establish intra-page relationships
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
        """Collects evidence and generates multi-source structure hypotheses for a region."""
        text = reg.text.strip()
        bbox = reg.bbox

        hypotheses: List[StructureHypothesis] = []

        # 1. PARSER HYPOTHESIS (Text-pattern based)
        parser_ev: List[DocumentEvidence] = []
        parser_type: DocumentRegionType = "UNKNOWN"
        parser_conf = 0.5

        q_num_match = re.search(
            r"^(?:Q(?:uestion)?[\s\.\:]*)?(\d+|[IVXLCDM]+)[\.\:\)\s]+", text, re.IGNORECASE
        )
        subq_match = re.search(r"^\(?([a-z]|[ivxlcdm]+)\)[\.\:\s]+", text, re.IGNORECASE)
        interrogative_match = re.search(
            r"\b(what|why|how|explain|describe|calculate|evaluate|find|prove|derive|compare|define|list|state|discuss|show|write|determine|solve|select|choose|identify|mark|indicate)\b",
            text,
            re.IGNORECASE,
        )

        if q_num_match:
            parser_ev.append(
                DocumentEvidence(
                    signal_type="numbering_pattern",
                    description=f"Matched primary question numbering pattern '{q_num_match.group(0).strip()}'",
                    weight=0.4,
                    score=0.9,
                )
            )

        if subq_match and not q_num_match:
            parser_ev.append(
                DocumentEvidence(
                    signal_type="numbering_pattern",
                    description=f"Matched subquestion numbering pattern '{subq_match.group(0).strip()}'",
                    weight=0.4,
                    score=0.85,
                )
            )

        if interrogative_match:
            parser_ev.append(
                DocumentEvidence(
                    signal_type="question_interrogative",
                    description=f"Contains question interrogative keyword '{interrogative_match.group(1)}'",
                    weight=0.3,
                    score=0.8,
                )
            )

        opt_match = re.search(
            r"^\(?[A-Da-d1-9i-zIVXLCDM]+\)[\.\:\s]+|^(?:Option|Choice)\s+[\(]?[A-Za-z0-9ivxlcdm]+\)?[\.\:\s]*",
            text,
            re.IGNORECASE,
        )
        if opt_match and not interrogative_match:
            parser_ev.append(
                DocumentEvidence(
                    signal_type="option_formatting",
                    description=f"Matched option pattern '{opt_match.group(0).strip()}'",
                    weight=0.5,
                    score=0.95,
                )
            )

        inst_match = re.search(
            r"^\s*(?:Note|Instructions|General Instructions|Notice|Read carefully|Answer all|All questions carry)[\s\:\-\.]",
            text,
            re.IGNORECASE,
        )
        if inst_match:
            parser_ev.append(
                DocumentEvidence(
                    signal_type="section_formatting",
                    description=f"Matched instruction keyword pattern '{inst_match.group(0).strip()}'",
                    weight=0.5,
                    score=0.9,
                )
            )

        sec_match = re.search(
            r"^\s*(?:SECTION|PART|GROUP|UNIT|CHAPTER)\s+[\-–\s]*[A-Z0-9IVX]+", text, re.IGNORECASE
        )
        if sec_match:
            parser_ev.append(
                DocumentEvidence(
                    signal_type="heading_formatting",
                    description=f"Matched section header pattern '{sec_match.group(0).strip()}'",
                    weight=0.5,
                    score=0.95,
                )
            )

        if sec_match:
            parser_type = "SECTION_HEADER"
            parser_conf = 0.95
        elif inst_match:
            parser_type = "INSTRUCTION"
            parser_conf = 0.90
        elif opt_match and not interrogative_match:
            parser_type = "OPTION"
            parser_conf = 0.88
        elif q_num_match and interrogative_match:
            parser_type = "QUESTION"
            parser_conf = 0.92
        elif q_num_match:
            parser_type = "QUESTION"
            parser_conf = 0.80
        elif subq_match:
            parser_type = "SUBQUESTION"
            parser_conf = 0.82
        elif interrogative_match:
            parser_type = "QUESTION"
            parser_conf = 0.70

        if parser_type != "UNKNOWN":
            hypotheses.append(
                StructureHypothesis(
                    region_id=reg.region_id,
                    hypothesized_type=parser_type,
                    confidence=parser_conf,
                    source="parser",
                    evidence=parser_ev,
                )
            )

        # 2. LAYOUT ANALYZER HYPOTHESIS (Geometry & Position based)
        layout_ev: List[DocumentEvidence] = []
        layout_type: DocumentRegionType = "UNKNOWN"
        layout_conf = 0.50

        rel_top = bbox.y / page_height if page_height > 0 else 0
        rel_bottom = (bbox.y + bbox.height) / page_height if page_height > 0 else 0

        if rel_top < 0.06:
            layout_ev.append(
                DocumentEvidence(
                    signal_type="page_position",
                    description=f"Positioned near top margin (rel_y={rel_top:.3f})",
                    weight=0.4,
                    score=0.85,
                )
            )
            layout_type = "HEADER"
            layout_conf = 0.85
        elif rel_bottom > 0.94:
            layout_ev.append(
                DocumentEvidence(
                    signal_type="page_position",
                    description=f"Positioned near bottom margin (rel_y={rel_bottom:.3f})",
                    weight=0.4,
                    score=0.85,
                )
            )
            layout_type = "FOOTER"
            layout_conf = 0.85

        if reg.metadata.get("role") == "visual_element" or re.search(
            r"\[(?:Diagram|Figure|Image|Graph)\]|\b(?:Fig\.|Figure|Diagram)\s*\d+", text, re.IGNORECASE
        ):
            layout_ev.append(
                DocumentEvidence(
                    signal_type="table_geometry",
                    description="Visual element / Diagram reference detected in block metadata",
                    weight=0.5,
                    score=0.9,
                )
            )
            layout_type = "DIAGRAM"
            layout_conf = 0.90
        elif re.search(r"^\|.*?\|", text, re.MULTILINE) or re.search(r"\bTable\s*\d*[\:\.\s]", text, re.IGNORECASE) or (
            "\t" in text and text.count("\t") >= 2
        ):
            layout_ev.append(
                DocumentEvidence(
                    signal_type="table_geometry",
                    description="Tabular syntax / grid structure detected",
                    weight=0.5,
                    score=0.88,
                )
            )
            layout_type = "TABLE"
            layout_conf = 0.88

        if layout_type != "UNKNOWN":
            hypotheses.append(
                StructureHypothesis(
                    region_id=reg.region_id,
                    hypothesized_type=layout_type,
                    confidence=layout_conf,
                    source="layout_analyzer",
                    evidence=layout_ev,
                )
            )

        # 3. SEMANTIC ANALYZER HYPOTHESIS
        sem_ev: List[DocumentEvidence] = []
        sem_type: DocumentRegionType = "UNKNOWN"
        sem_conf = 0.50

        if re.search(r"\b(?:Roll No|Name|Date|Subject|Time Allowed|Maximum Marks|Class|Session)\b", text, re.IGNORECASE):
            sem_ev.append(
                DocumentEvidence(
                    signal_type="semantic_signal",
                    description="Document metadata header terms detected",
                    weight=0.5,
                    score=0.9,
                )
            )
            sem_type = "METADATA"
            sem_conf = 0.90
            hypotheses.append(
                StructureHypothesis(
                    region_id=reg.region_id,
                    hypothesized_type=sem_type,
                    confidence=sem_conf,
                    source="semantic_analyzer",
                    evidence=sem_ev,
                )
            )

        # 4. RESOLVE / CONSOLIDATE HYPOTHESES
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
        current_question_id: Optional[str] = None

        for idx, reg in enumerate(sorted_regs):
            if idx > 0:
                prev_reg = sorted_regs[idx - 1]
                rels.append(
                    RegionRelationship(
                        source_region_id=reg.region_id,
                        target_region_id=prev_reg.region_id,
                        relationship_type="follows",
                        confidence=0.95,
                        evidence=[
                            DocumentEvidence(
                                signal_type="spatial_position",
                                description=f"Sequentially follows region '{prev_reg.region_id}' in reading order",
                                weight=0.3,
                                score=0.95,
                            )
                        ],
                    )
                )

            if reg.region_type == "QUESTION":
                current_question_id = reg.region_id
            elif reg.region_type in ("OPTION", "SUBQUESTION", "TABLE", "DIAGRAM", "TABLE_CELL"):
                if current_question_id:
                    rels.append(
                        RegionRelationship(
                            source_region_id=current_question_id,
                            target_region_id=reg.region_id,
                            relationship_type="contains",
                            confidence=0.90,
                            evidence=[
                                DocumentEvidence(
                                    signal_type="surrounding_regions",
                                    description=f"Question '{current_question_id}' spatially contains {reg.region_type.lower()} region '{reg.region_id}'",
                                    weight=0.4,
                                    score=0.90,
                                )
                            ],
                        )
                    )
                    rels.append(
                        RegionRelationship(
                            source_region_id=reg.region_id,
                            target_region_id=current_question_id,
                            relationship_type="belongs_to",
                            confidence=0.90,
                            evidence=[
                                DocumentEvidence(
                                    signal_type="surrounding_regions",
                                    description=f"Region '{reg.region_id}' belongs to parent question '{current_question_id}'",
                                    weight=0.4,
                                    score=0.90,
                                )
                            ],
                        )
                    )

                    reg.parent_region_id = current_question_id
                    parent_reg = next((r for r in regions if r.region_id == current_question_id), None)
                    if parent_reg and reg.region_id not in parent_reg.child_region_ids:
                        parent_reg.child_region_ids.append(reg.region_id)

        for reg in regions:
            if reg.region_type == "OPTION" and reg.parent_region_id:
                peer_options = [
                    r for r in regions if r.region_type == "OPTION" and r.parent_region_id == reg.parent_region_id and r.region_id != reg.region_id
                ]
                for peer in peer_options:
                    rels.append(
                        RegionRelationship(
                            source_region_id=reg.region_id,
                            target_region_id=peer.region_id,
                            relationship_type="same_structure_as",
                            confidence=0.85,
                            evidence=[
                                DocumentEvidence(
                                    signal_type="repeated_layout_pattern",
                                    description=f"Peer option under common question '{reg.parent_region_id}'",
                                    weight=0.3,
                                    score=0.85,
                                )
                            ],
                        )
                    )

        return rels

    def _extract_cross_page_relationships(
        self, pages: List[DocumentPage]
    ) -> List[RegionRelationship]:
        """Extracts relationships across pages (e.g., continuation_of)."""
        cross_rels: List[RegionRelationship] = []
        if len(pages) < 2:
            return cross_rels

        for i in range(len(pages) - 1):
            curr_page = pages[i]
            next_page = pages[i + 1]

            if not curr_page.regions or not next_page.regions:
                continue

            last_reg = curr_page.regions[-1]
            first_reg = next_page.regions[0]

            last_text = last_reg.text.strip()
            if last_text and not last_text[-1] in (".", "?", "!", ":", ";"):
                cross_rels.append(
                    RegionRelationship(
                        source_region_id=first_reg.region_id,
                        target_region_id=last_reg.region_id,
                        relationship_type="continuation_of",
                        confidence=0.75,
                        evidence=[
                            DocumentEvidence(
                                signal_type="continuation_relationship",
                                description=f"Region '{first_reg.region_id}' on page {next_page.page_number} continues region '{last_reg.region_id}' from page {curr_page.page_number}",
                                weight=0.4,
                                score=0.75,
                            )
                        ],
                    )
                )

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
                    reg.evidence.append(
                        DocumentEvidence(
                            signal_type="semantic_signal",
                            description="Dense vector embedding attached from Step 8 embedding engine",
                            weight=0.2,
                            score=1.0,
                        )
                    )
        except Exception as e:
            print(f"[DocumentUnderstandingService] Embedding attachment notice: {e}")


def get_debug_summary(result: DocumentUnderstandingResult) -> Dict[str, Any]:
    """
    Developer / debug representation helper for inspecting document understanding state.
    Exposes region hypotheses, evidence sources, verification states, and cost accounting.
    """
    region_summaries = []
    for r in result.regions:
        region_summaries.append(
            {
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
            }
        )

    return {
        "document_id": result.document_id,
        "vlm_status": result.vlm_status,
        "total_regions": len(result.regions),
        "total_relationships": len(result.relationships),
        "conflicts_count": len(result.conflicts),
        "verification_summary": result.verification_summary,
        "cost_accounting": result.cost_accounting.model_dump() if result.cost_accounting else None,
        "regions": region_summaries,
    }
