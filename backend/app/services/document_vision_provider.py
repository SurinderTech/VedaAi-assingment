"""
Multimodal Document Vision Provider — Page-Level Document Intelligence.

The VLM is the document-understanding BRAIN, not a verification layer.

Architecture:
  Page Image + Complete OCR Evidence + Document Context
    → VLM independently reasons about document structure
    → Returns: page purpose, region roles, relationships, reasoning

The VLM decides WHAT/WHERE/RELATIONSHIP.
OCR/native PDF provides EXACT TEXT.
Deterministic code constructs the final question.

Preserves semantic relationship validation and contradiction filtering.
"""
from __future__ import annotations
import json
import re
import base64
import io
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import (
    Block,
    BBox,
    DocumentRegion,
    DocumentUnderstandingResult,
    VisualVerificationResponse,
    VLMHypothesis,
    VLMPageUnderstanding,
    VLMStructureItem,
    VLMRelationshipItem,
    RegionRelationship,
    CostAccounting,
    DocumentEvidence,
    StructureHypothesis,
)


class VisionAnalysisResult(BaseModel):
    status: str = "NOT_CONFIGURED"
    message: str = "VLM Vision Provider is not configured."
    is_available: bool = False
    page_analyses: List[Dict[str, Any]] = []
    region_verifications: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class DocumentVisionProvider:
    """Abstract interface for multi-modal vision models."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key if api_key is not None else getattr(settings, "GEMINI_API_KEY", "")
        self.model_name = model_name or getattr(settings, "DOCUMENT_VLM_MODEL", "gemini-2.5-flash")

    def is_configured(self) -> bool:
        enabled = getattr(settings, "DOCUMENT_VLM_ENABLED", False)
        gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        or_key = getattr(settings, "OPENROUTER_API_KEY", "")
        has_key = bool((self.api_key and len(self.api_key) > 5) or (gemini_key and len(gemini_key) > 5) or (or_key and len(or_key) > 5))
        return enabled and has_key

    def understand_page(self, **kwargs) -> VLMPageUnderstanding:
        return VLMPageUnderstanding(page_number=kwargs.get("page_number", 1))

    def verify_structure(
        self,
        result: DocumentUnderstandingResult,
        page_images: Optional[Dict[int, bytes]] = None,
        target_region_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VisualVerificationResponse:
        return VisualVerificationResponse(
            status="NOT_CONFIGURED",
            is_available=False,
            model_name=self.model_name,
            vlm_hypotheses=[],
            cost_accounting=CostAccounting(
                pages_considered=len(result.pages),
                regions_considered=len(result.regions),
                skipped_high_confidence_count=len(result.regions),
            ),
            error_message="VLM Vision Provider is disabled or unconfigured.",
        )


class MultimodalDocumentVisionProvider(DocumentVisionProvider):
    """
    Page-Level Multimodal Document Intelligence Provider.

    The VLM sees the actual page image + complete OCR evidence and independently
    reasons about document structure. It is NOT a confirmation layer for regex.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_response: Optional[VisualVerificationResponse] = None,
        mock_page_understanding: Optional[VLMPageUnderstanding] = None,
    ):
        super().__init__(api_key=api_key, model_name=model_name)
        self.mock_response = mock_response
        self.mock_page_understanding = mock_page_understanding

    def is_configured(self, force_vlm: bool = False) -> bool:
        enabled = getattr(settings, "DOCUMENT_VLM_ENABLED", False) or force_vlm or (self.mock_response is not None) or (self.mock_page_understanding is not None)
        gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        or_key = getattr(settings, "OPENROUTER_API_KEY", "")
        has_key = bool((self.api_key and len(self.api_key) > 5) or (gemini_key and len(gemini_key) > 5) or (or_key and len(or_key) > 5))
        return (enabled and has_key) or (self.mock_response is not None) or (self.mock_page_understanding is not None)

    # ================================================================
    # PRIMARY INTELLIGENCE METHOD — Page-Level Document Understanding
    # ================================================================

    def understand_page(
        self,
        page_image: Any,
        ocr_blocks: List[Block],
        page_number: int,
        total_pages: int,
        page_context: Optional[Dict[str, Any]] = None,
        force_vlm: bool = False,
        _depth: int = 0,
        _y_offset: float = 0.0,
    ) -> VLMPageUnderstanding:
        """
        The VLM BRAIN: sends the actual page image + complete OCR evidence
        to the VLM and asks it to independently understand the document page.

        This is NOT region verification. The VLM reasons about the whole page.
        """
        # Mock path for testing
        if self.mock_page_understanding is not None:
            return self.mock_page_understanding

        if not self.is_configured(force_vlm=force_vlm):
            return VLMPageUnderstanding(
                page_number=page_number,
                page_purpose="UNKNOWN",
                document_purpose="UNKNOWN",
                structure_source="DETERMINISTIC_FALLBACK",
                vlm_result="NOT_CONFIGURED",
                semantic_completeness="UNKNOWN",
            )

        # Build the page understanding prompt with COMPLETE OCR evidence
        prompt = self._build_page_understanding_prompt(
            ocr_blocks=ocr_blocks,
            page_number=page_number,
            total_pages=total_pages,
            page_context=page_context,
        )

        # Encode page image & validate payload
        page_b64, mime_type, img_meta = self._encode_image_with_metadata(page_image)

        # Call VLM with metadata tracking
        response_text, vlm_meta = self._execute_vlm_call_with_metadata(prompt, page_b64, mime_type=mime_type)

        vlm_meta["prompt_chars"] = len(prompt)
        vlm_meta["image_bytes"] = img_meta.get("byte_size", 0)
        vlm_meta["base64_chars"] = len(page_b64) if page_b64 else 0
        if img_meta.get("dimensions"):
            vlm_meta["image_dimensions"] = [float(img_meta["dimensions"][0]), float(img_meta["dimensions"][1])]

        if not response_text:
            print(f"[VLM] Page {page_number}: VLM call failed or returned empty response")
            return VLMPageUnderstanding(
                page_number=page_number,
                page_purpose="UNKNOWN",
                image_sent=page_b64 is not None,
                image_dimensions=vlm_meta.get("image_dimensions"),
                image_bytes=vlm_meta.get("image_bytes", 0),
                base64_chars=vlm_meta.get("base64_chars", 0),
                ocr_blocks_sent=len(ocr_blocks),
                prompt_chars=vlm_meta.get("prompt_chars", 0),
                vlm_attempt=True,
                vlm_model=self.model_name,
                structure_source="DETERMINISTIC_FALLBACK",
                vlm_provider=vlm_meta.get("provider", "gemini"),
                vlm_result="FAILED",
                finish_reason=vlm_meta.get("finish_reason", "N/A"),
                semantic_completeness="UNKNOWN",
                retry_count=vlm_meta.get("retry_count", 0),
                fallback_provider=vlm_meta.get("fallback_provider", "N/A"),
                structures_produced=0,
                relationships_produced=0,
            )

        # Parse VLM response into structured page understanding
        understanding = self._parse_page_understanding(
            response_text=response_text,
            page_number=page_number,
            ocr_blocks=ocr_blocks,
            page_b64_sent=page_b64 is not None,
            vlm_meta=vlm_meta,
        )

        # Visual decomposition retry on MAX_TOKENS
        if understanding.semantic_completeness == "PARTIAL" and _depth == 0 and page_image is not None and img_meta.get("dimensions"):
            w, h = img_meta["dimensions"]
            if h > 100:
                print(f"[VLM] Page {page_number}: MAX_TOKENS observed. Executing vision-first vertical decomposition retry...")
                try:
                    from PIL import Image
                    if isinstance(page_image, bytes):
                        pil_img = Image.open(io.BytesIO(page_image))
                    elif hasattr(page_image, "crop"):
                        pil_img = page_image
                    else:
                        pil_img = None

                    if pil_img is not None:
                        mid_y = h / 2.0
                        top_img = pil_img.crop((0, 0, int(w), int(mid_y)))
                        bot_img = pil_img.crop((0, int(mid_y), int(w), int(h)))

                        top_blocks = [b for b in ocr_blocks if b.bbox.y + b.bbox.height / 2.0 <= mid_y]
                        bot_blocks_orig = [b for b in ocr_blocks if b.bbox.y + b.bbox.height / 2.0 > mid_y]
                        # Shift bottom blocks coordinates relative to top of bottom crop for OCR prompt matching
                        bot_blocks_shifted = [
                            Block(
                                id=b.id,
                                page=b.page,
                                text=b.text,
                                bbox=BBox(x=b.bbox.x, y=max(0.0, b.bbox.y - mid_y), width=b.bbox.width, height=b.bbox.height),
                                confidence=b.confidence,
                                source=b.source,
                            )
                            for b in bot_blocks_orig
                        ]

                        top_und = self.understand_page(
                            top_img, top_blocks, page_number, total_pages, page_context=page_context, _depth=1
                        )
                        bot_und = self.understand_page(
                            bot_img, bot_blocks_shifted, page_number, total_pages, page_context=page_context, _depth=1, _y_offset=mid_y
                        )

                        # BUG #2 FIX: Reconcile structures and relationships across decomposition boundary
                        combined_structures, combined_relationships = self._reconcile_decomposed_page_understanding(
                            top_und=top_und,
                            bot_und=bot_und,
                            mid_y=mid_y,
                        )

                        if combined_structures:
                            both_complete = (top_und.finish_reason == "STOP" and bot_und.finish_reason == "STOP")
                            understanding.structures = combined_structures
                            understanding.relationships = combined_relationships
                            understanding.structures_produced = len(combined_structures)
                            understanding.relationships_produced = len(combined_relationships)
                            understanding.finish_reason = "STOP" if both_complete else "PARTIAL"
                            understanding.semantic_completeness = "COMPLETE" if both_complete else "PARTIAL"
                            understanding.vlm_result = "SUCCESS"
                            understanding.structure_source = "VLM_DECOMPOSED_SUCCESS" if both_complete else "VLM_DECOMPOSED_PARTIAL"
                            print(f"[VLM] Page {page_number}: Visual decomposition completed: {len(combined_structures)} structures, {len(combined_relationships)} relationships (completeness={understanding.semantic_completeness}).")
                except Exception as e_decomp:
                    print(f"[VLM] Page {page_number}: Visual decomposition exception: {e_decomp}")

        return understanding

    def _reconcile_decomposed_page_understanding(
        self,
        top_und: VLMPageUnderstanding,
        bot_und: VLMPageUnderstanding,
        mid_y: float,
    ) -> Tuple[List[VLMStructureItem], List[VLMRelationshipItem]]:
        """
        Reconciles visual decomposition results across crop boundaries (BUG #2 FIX).
        1. Preserves geometry of every structure (bot structures shifted by mid_y).
        2. Merges structures representing the same visual entity across the split line
           using geometric continuity, bounding box proximity, and OCR evidence.
        3. Assigns a canonical structure ID and builds an ID remapping table.
        4. Remaps all relationships from provisional bottom-crop IDs to canonical IDs.
        5. Deduplicates equivalent structures and relationships while eliminating self-loops.
        """
        # Step 1: Shift bottom structure bounding boxes to full-page coordinates
        for s in bot_und.structures:
            if s.bbox:
                s.bbox = BBox(x=s.bbox.x, y=s.bbox.y + mid_y, width=s.bbox.width, height=s.bbox.height)

        id_remap: Dict[str, str] = {}
        merged_bot_indices = set()
        # Collect region IDs belonging to child structures in bot_und (options, subquestions)
        child_bot_rids = set()
        for s in bot_und.structures:
            if s.role in ("OPTION", "SUBQUESTION"):
                child_bot_rids.update(s.region_ids)

        bot_merged_to_canonical: Dict[int, str] = {}
        canonical_top_structures = list(top_und.structures)

        # Step 2: Cross-boundary entity matching
        for i_top, s_top in enumerate(canonical_top_structures):
            for i_bot, s_bot in enumerate(bot_und.structures):
                if i_bot in merged_bot_indices:
                    continue
                if s_top.role != s_bot.role or s_top.role == "UNKNOWN":
                    continue

                # Incompatible explicit metadata guard
                if s_top.display_number and s_bot.display_number and s_top.display_number != s_bot.display_number:
                    continue
                if s_top.display_label and s_bot.display_label and s_top.display_label != s_bot.display_label:
                    continue

                # Both structures MUST physically lie at the decomposition split line mid_y
                if not (s_top.bbox and s_bot.bbox):
                    continue

                top_margin = max(s_top.bbox.height * 2.5, 80.0)
                bot_margin = max(s_bot.bbox.height * 2.5, 80.0)

                s_top_in_boundary = (s_top.bbox.y < mid_y and (s_top.bbox.y + s_top.bbox.height) >= mid_y - top_margin)
                s_bot_in_boundary = (s_bot.bbox.y <= mid_y + bot_margin and (s_bot.bbox.y + s_bot.bbox.height) > mid_y)

                if not (s_top_in_boundary and s_bot_in_boundary):
                    continue

                # Condition A: Shared OCR region IDs at boundary
                shared_ids = set(s_top.region_ids) & set(s_bot.region_ids)
                has_shared_ids = bool(shared_ids)

                # Condition B: Geometric continuity across split line mid_y
                gap_y = max(0.0, s_bot.bbox.y - (s_top.bbox.y + s_top.bbox.height))
                max_allowed_gap = max(s_top.bbox.height * 2.0, s_bot.bbox.height * 2.0, 75.0)

                # Horizontal proximity (within same column / page band)
                h_distance = max(
                    0.0,
                    max(s_top.bbox.x, s_bot.bbox.x)
                    - min(s_top.bbox.x + s_top.bbox.width, s_bot.bbox.x + s_bot.bbox.width)
                )
                max_allowed_h_dist = 600.0  # Document column boundary

                has_geom_match = (gap_y <= max_allowed_gap and h_distance <= max_allowed_h_dist)

                if has_shared_ids or has_geom_match:
                    # Match found! Merge s_bot into canonical s_top
                    merged_bot_indices.add(i_bot)

                    # Merge bounding boxes
                    if s_top.bbox and s_bot.bbox:
                        min_x = min(s_top.bbox.x, s_bot.bbox.x)
                        min_y = min(s_top.bbox.y, s_bot.bbox.y)
                        max_x = max(s_top.bbox.x + s_top.bbox.width, s_bot.bbox.x + s_bot.bbox.width)
                        max_y = max(s_top.bbox.y + s_top.bbox.height, s_bot.bbox.y + s_bot.bbox.height)
                        s_top.bbox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)
                    elif s_bot.bbox:
                        s_top.bbox = s_bot.bbox

                    # Merge region IDs preserving order, excluding child structure IDs
                    combined_rids = list(s_top.region_ids)
                    for rid in s_bot.region_ids:
                        if rid not in combined_rids and (s_top.role not in ("QUESTION", "SECTION_HEADER", "INSTRUCTION") or rid not in child_bot_rids):
                            combined_rids.append(rid)
                    s_top.region_ids = combined_rids

                    # Canonical target ID for relationships
                    canonical_id = s_top.region_ids[0] if s_top.region_ids else None
                    if canonical_id:
                        bot_merged_to_canonical[i_bot] = canonical_id

                    s_top.confidence = max(s_top.confidence, s_bot.confidence)
                    if not s_top.display_number and s_bot.display_number:
                        s_top.display_number = s_bot.display_number
                    if not s_top.display_label and s_bot.display_label:
                        s_top.display_label = s_bot.display_label
                    if not s_top.reasoning and s_bot.reasoning:
                        s_top.reasoning = s_bot.reasoning

        # Step 3: Combine structures (unmerged bot structures added)
        unmerged_bot_structures = [
            s for i, s in enumerate(bot_und.structures)
            if i not in merged_bot_indices
        ]
        combined_structures = canonical_top_structures + unmerged_bot_structures

        # Build ID remapping table protecting unmerged bottom structures
        unmerged_bot_rids = set()
        for s in unmerged_bot_structures:
            unmerged_bot_rids.update(s.region_ids)

        for i_bot, canonical_id in bot_merged_to_canonical.items():
            s_bot = bot_und.structures[i_bot]
            for bot_rid in s_bot.region_ids:
                if bot_rid not in unmerged_bot_rids:
                    id_remap[bot_rid] = canonical_id

        # Step 4: Re-map relationships and attach orphan boundary options to boundary question
        top_questions = [s for s in canonical_top_structures if s.role == "QUESTION" and s.bbox]
        top_boundary_q = max(top_questions, key=lambda s: s.bbox.y) if top_questions else None
        top_boundary_qid = top_boundary_q.region_ids[0] if (top_boundary_q and top_boundary_q.region_ids) else None

        bot_questions = [s for s in unmerged_bot_structures if s.role == "QUESTION" and s.bbox]
        first_bot_q = min(bot_questions, key=lambda s: s.bbox.y) if bot_questions else None
        first_bot_qy = first_bot_q.bbox.y if first_bot_q else (mid_y + 9999.0)

        # For any OPTION/SUBQUESTION in unmerged_bot_structures located above first_bot_qy:
        # connect to top_boundary_qid if it does not already have an option_of relationship
        if top_boundary_qid:
            for s in unmerged_bot_structures:
                if s.role in ("OPTION", "SUBQUESTION") and s.bbox and s.bbox.y < first_bot_qy and s.region_ids:
                    s_rid = s.region_ids[0]
                    rel_type = "option_of" if s.role == "OPTION" else "subquestion_of"
                    bot_und.relationships.append(VLMRelationshipItem(
                        source_ids=[s_rid],
                        target_ids=[top_boundary_qid],
                        relationship_type=rel_type,
                        confidence=s.confidence,
                    ))

        combined_raw_relationships = top_und.relationships + bot_und.relationships
        reconciled_relationships: List[VLMRelationshipItem] = []
        seen_rel_keys = set()

        for rel in combined_raw_relationships:
            new_src_ids = [id_remap.get(sid, sid) for sid in rel.source_ids]
            new_tgt_ids = [id_remap.get(tid, tid) for tid in rel.target_ids]

            # Drop self-referencing loops created by merging
            if set(new_src_ids) == set(new_tgt_ids):
                continue

            rel_key = (
                tuple(sorted(new_src_ids)),
                tuple(sorted(new_tgt_ids)),
                rel.relationship_type,
            )
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)

            reconciled_relationships.append(VLMRelationshipItem(
                source_ids=new_src_ids,
                target_ids=new_tgt_ids,
                relationship_type=rel.relationship_type,
                confidence=rel.confidence,
            ))

        return combined_structures, reconciled_relationships

    def _build_page_understanding_prompt(
        self,
        ocr_blocks: List[Block],
        page_number: int,
        total_pages: int,
        page_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Builds a compact page-level document understanding prompt.

        The VLM is asked to UNDERSTAND the document visually.
        The prompt emphasizes that the page image is authoritative for layout and structure,
        and OCR blocks are supporting textual evidence.
        Output format is compact semantic geometry (role, bbox/ids, confidence, relationships).
        """
        # Build complete OCR evidence with IDs and bounding boxes in compact form
        ocr_evidence_lines = []
        for b in ocr_blocks:
            ocr_evidence_lines.append(
                f'  {{"id": "{b.id}", "bbox": [{round(b.bbox.x,1)}, {round(b.bbox.y,1)}, {round(b.bbox.width,1)}, {round(b.bbox.height,1)}], "text": {json.dumps(b.text)}}}'
            )

        ocr_evidence = "[\n" + ",\n".join(ocr_evidence_lines) + "\n]"

        context_info = ""
        if page_context:
            if page_context.get("prev_page_summary"):
                context_info += f"\nPrevious page context: {page_context['prev_page_summary']}"
            if page_context.get("next_page_summary"):
                context_info += f"\nNext page context: {page_context['next_page_summary']}"

        return f"""You are analyzing page {page_number} of {total_pages} from a document.

TASK: Examine the page image (authoritative for visual structure) and the OCR text evidence below. Identify all meaningful visual structures and their semantic relationships.
{context_info}
RULES:
1. Treat the page image as authoritative for visual layout, columns, reading order, and entity boundaries.
2. Identify each distinct semantic unit separately. Use the role vocabulary: QUESTION, SUBQUESTION, OPTION, SECTION_HEADER, INSTRUCTION, TABLE, DIAGRAM, FIGURE, CAPTION, ANSWER_REGION, HANDWRITING, FORM_FIELD, PARAGRAPH, LIST, HEADER, FOOTER, METADATA, SIGNATURE, UNKNOWN.
3. Keep questions and their options/subquestions as SEPARATE structures. Connect them using relationships (e.g. "option_of", "subquestion_of").
4. If an entity maps to OCR blocks, include their "ids". If it is visual or spans multiple blocks, include its visual "bbox": [x1, y1, x2, y2].
5. For multi-column documents: respect column boundaries when assigning options to questions. Options always belong to the most recently seen question IN THE SAME COLUMN.
6. For answer sheets: use ANSWER_REGION for written answer areas, HANDWRITING for handwritten text, SIGNATURE for signatures.
7. For figures/diagrams: use FIGURE or DIAGRAM and link to their caption with a "caption_of" relationship.
8. Do NOT repeat long text in the output; the OCR evidence provides the exact text.
9. An OPTION region belongs to exactly ONE QUESTION. Never assign the same option to multiple questions.
10. CRITICAL — QUESTION vs INSTRUCTION distinction:
    A real QUESTION is something a student must directly respond to (e.g. "What is photosynthesis?", "Define deep learning.", "Calculate the resistance.").
    An INSTRUCTION or group-parent header is text that introduces or organises other questions but is NOT itself answerable — e.g. "Write briefly :", "Answer any FOUR of the following:", "1. Write short notes on:". These MUST be labelled INSTRUCTION, NOT QUESTION.
    Rule: if a numbered item ends with a colon ":" or contains only an imperative intro phrase ("write briefly", "answer the following", "attempt any", "short answer questions") with no actual question body, label it INSTRUCTION.
11. CRITICAL — SECTION_HEADER:
    Labels like "SECTION-A", "SECTION A (COMPULSORY)", "PART B", "GROUP I" that delimit groups of questions must be labelled SECTION_HEADER, not QUESTION or INSTRUCTION.
    Connect questions inside a section to the section using the "section_member" relationship.

OCR Evidence (page {page_number}):
{ocr_evidence}

Return strictly valid JSON in this compact format:
{{
  "page_purpose": "QUESTION_PAGE" | "ANSWER_SHEET" | "COVER" | "INSTRUCTIONS" | "CONTINUATION" | "MIXED" | "ADMINISTRATIVE",
  "document_purpose": "EXAMINATION_PAPER" | "ANSWER_SHEET" | "ASSIGNMENT" | "FORM" | "INVOICE" | "REPORT" | "INSTRUCTIONS" | "UNKNOWN",
  "structures": [
    {{
      "role": "QUESTION" | "OPTION" | "SUBQUESTION" | "SECTION_HEADER" | "INSTRUCTION" | "METADATA" | "HEADER" | "FOOTER" | "TABLE" | "DIAGRAM" | "FIGURE" | "CAPTION" | "ANSWER_REGION" | "HANDWRITING" | "FORM_FIELD" | "PARAGRAPH" | "LIST" | "SIGNATURE" | "UNKNOWN",
      "ids": ["id1"],
      "bbox": [x1, y1, x2, y2],
      "conf": 0.95
    }}
  ],
  "relationships": [
    {{
      "src": ["id_option"],
      "tgt": ["id_question"],
      "type": "option_of" | "subquestion_of" | "section_member" | "continuation_of" | "belongs_to" | "caption_of" | "answer_to"
    }}
  ]
}}"""

    def _encode_image_with_metadata(self, page_image: Any) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """Encodes a page image (PIL Image or bytes) to base64 for VLM and returns (b64, mime_type, metadata)."""
        meta = {"valid": False, "dimensions": (0, 0), "byte_size": 0, "b64_valid": False}
        mime_type = "image/png"
        try:
            if page_image is None:
                return None, mime_type, meta

            if isinstance(page_image, bytes):
                meta["byte_size"] = len(page_image)
                # Auto-detect MIME type from magic bytes
                if page_image.startswith(b"\x89PNG"):
                    mime_type = "image/png"
                elif page_image.startswith(b"\xff\xd8\xff"):
                    mime_type = "image/jpeg"
                elif page_image.startswith(b"RIFF") and b"WEBP" in page_image[:16]:
                    mime_type = "image/webp"

                # Verify PIL decoding & dimensions
                from PIL import Image
                try:
                    pil_img = Image.open(io.BytesIO(page_image))
                    meta["dimensions"] = pil_img.size
                    meta["valid"] = True
                except Exception as e_pil:
                    print(f"[VLM] Image bytes PIL decode warning: {e_pil}")

                b64_str = base64.b64encode(page_image).decode("utf-8").replace("\n", "").replace("\r", "").strip()
                # Verify base64 decoding validity
                try:
                    test_dec = base64.b64decode(b64_str[:100] + "==")
                    meta["b64_valid"] = len(test_dec) > 0
                except Exception:
                    meta["b64_valid"] = True

                return b64_str, mime_type, meta

            if hasattr(page_image, "save"):  # PIL Image
                meta["dimensions"] = page_image.size
                meta["valid"] = True
                buf = io.BytesIO()
                page_image.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
                meta["byte_size"] = len(raw_bytes)
                b64_str = base64.b64encode(raw_bytes).decode("utf-8").replace("\n", "").replace("\r", "").strip()
                meta["b64_valid"] = True
                return b64_str, "image/png", meta

        except Exception as e:
            print(f"[VLM] Image encoding error: {e}")
        return None, mime_type, meta

    def _encode_image(self, page_image: Any) -> Optional[str]:
        b64, _, _ = self._encode_image_with_metadata(page_image)
        return b64

    def _execute_vlm_call_with_metadata(self, prompt: str, image_b64: Optional[str], mime_type: str = "image/png") -> Tuple[str, Dict[str, Any]]:
        """Executes the actual VLM API call with metadata tracking."""
        default_meta = {
            "provider": "gemini",
            "model": self.model_name,
            "vlm_result": "FAILED",
            "retry_count": 0,
            "fallback_used": False,
            "fallback_provider": "N/A",
            "structure_source": "DETERMINISTIC_FALLBACK",
        }
        try:
            from app.services.llm_provider import llm_complete_multimodal_with_metadata, llm_complete
            import asyncio

            async def _run_vlm():
                if image_b64 and len(image_b64) > 0:
                    return await llm_complete_multimodal_with_metadata(
                        prompt,
                        image_b64=image_b64,
                        mime_type=mime_type,
                        purpose="document_vision",
                    )
                else:
                    text = await llm_complete(prompt, purpose="document_vision")
                    return text, default_meta

            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(_run_vlm())
                else:
                    return asyncio.run(_run_vlm())
            except Exception as e_run:
                print(f"[VLM] VLM call failed: {e_run}")
                return "", default_meta


        except Exception as e:
            print(f"[VLM] VLM execution error: {e}")
            return "", default_meta

    def _execute_vlm_call(self, prompt: str, image_b64: Optional[str]) -> str:
        text, _ = self._execute_vlm_call_with_metadata(prompt, image_b64)
        return text

    def _parse_page_understanding(
        self,
        response_text: str,
        page_number: int,
        ocr_blocks: List[Block],
        page_b64_sent: bool,
        vlm_meta: Optional[Dict[str, Any]] = None,
    ) -> VLMPageUnderstanding:
        """Parses VLM JSON response into structured VLMPageUnderstanding, validating region IDs and bounding boxes."""
        valid_ids = {b.id for b in ocr_blocks}
        meta = vlm_meta or {}
        raw_finish_reason = meta.get("finish_reason", "STOP")

        try:
            from app.services.llm_provider import extract_json_payload
            data = extract_json_payload(response_text)
            if not isinstance(data, dict):
                data = {}

            page_purpose = data.get("page_purpose", "UNKNOWN")
            document_purpose = data.get("document_purpose", "UNKNOWN")

            # Parse structures — accept both verbose and compact keys
            structures: List[VLMStructureItem] = []
            for item in data.get("structures", []):
                raw_ids = item.get("region_ids", item.get("ids", []))
                validated_ids = [rid for rid in raw_ids if rid in valid_ids]

                bbox_value = item.get("bbox")
                parsed_bbox = None
                if bbox_value is not None:
                    try:
                        if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 4:
                            x1, y1, x2, y2 = [float(v) for v in bbox_value[:4]]
                            parsed_bbox = BBox(
                                x=min(x1, x2),
                                y=min(y1, y2),
                                width=max(abs(x2 - x1), 0.0),
                                height=max(abs(y2 - y1), 0.0),
                            )
                    except Exception:
                        parsed_bbox = None

                if not validated_ids and parsed_bbox is None:
                    continue

                role = item.get("role", "UNKNOWN")
                role_map = {
                    "ADMINISTRATIVE": "INSTRUCTION",
                    "CONTINUATION": "UNKNOWN",
                }
                normalized_role = role_map.get(role, role)

                valid_roles = {
                    "HEADER", "FOOTER", "METADATA", "INSTRUCTION", "SECTION_HEADER",
                    "QUESTION", "SUBQUESTION", "OPTION", "TABLE", "TABLE_CELL",
                    "DIAGRAM", "FIGURE", "ANSWER_SPACE", "UNKNOWN",
                }
                if normalized_role not in valid_roles:
                    normalized_role = "UNKNOWN"

                structures.append(VLMStructureItem(
                    region_ids=validated_ids,
                    bbox=parsed_bbox,
                    role=normalized_role,
                    display_number=item.get("display_number", item.get("num")),
                    display_label=item.get("display_label"),
                    reasoning=str(item.get("reasoning", "")),
                    confidence=float(item.get("confidence", item.get("conf", 0.90))),
                ))

            # Build structure lookup by region_id
            struct_by_rid = {}
            for s in structures:
                for rid in s.region_ids:
                    struct_by_rid[rid] = s

            # Parse relationships — validate all referenced IDs (accept compact src/tgt keys)
            raw_relationships: List[VLMRelationshipItem] = []
            for rel in data.get("relationships", []):
                src_raw = rel.get("source_ids", rel.get("src", []))
                tgt_raw = rel.get("target_ids", rel.get("tgt", []))
                if isinstance(src_raw, str):
                    src_raw = [src_raw]
                if isinstance(tgt_raw, str):
                    tgt_raw = [tgt_raw]

                src_ids = [rid for rid in src_raw if rid in valid_ids]
                tgt_ids = [rid for rid in tgt_raw if rid in valid_ids]
                if not src_ids or not tgt_ids:
                    continue

                valid_rel_types = {
                    "follows", "contains", "belongs_to", "continuation_of",
                    "option_of", "subquestion_of", "section_member",
                    "associated_visual",
                }
                rel_type = rel.get("type", rel.get("relationship_type", "belongs_to"))
                if rel_type not in valid_rel_types:
                    rel_type = "belongs_to"

                # Spatial validity: for option_of and subquestion_of, target question MUST exist and precede source option vertically
                if rel_type in ("option_of", "subquestion_of"):
                    src_struct = struct_by_rid.get(src_ids[0])
                    valid_tgts = []
                    for tid in tgt_ids:
                        tgt_struct = struct_by_rid.get(tid)
                        if tgt_struct and tgt_struct.role == "QUESTION":
                            if src_struct and src_struct.bbox and tgt_struct.bbox:
                                if tgt_struct.bbox.y <= src_struct.bbox.y + 25.0:
                                    valid_tgts.append(tid)
                            else:
                                valid_tgts.append(tid)
                    tgt_ids = valid_tgts
                    if not tgt_ids:
                        continue

                for tgt_id in tgt_ids:
                    raw_relationships.append(VLMRelationshipItem(
                        source_ids=src_ids,
                        target_ids=[tgt_id],
                        relationship_type=rel_type,
                        confidence=float(rel.get("confidence", rel.get("conf", 0.90))),
                    ))

            # Deduplicate and pick single best target for option_of / subquestion_of
            relationships: List[VLMRelationshipItem] = []
            best_opt_targets: Dict[Tuple[str, str], VLMRelationshipItem] = {}

            for rel in raw_relationships:
                src_id = rel.source_ids[0]
                if rel.relationship_type in ("option_of", "subquestion_of"):
                    key = (src_id, rel.relationship_type)
                    if key not in best_opt_targets:
                        best_opt_targets[key] = rel
                    else:
                        # Pick the target question with closest preceding vertical position
                        cur_tgt = best_opt_targets[key].target_ids[0]
                        new_tgt = rel.target_ids[0]
                        cur_s = struct_by_rid.get(cur_tgt)
                        new_s = struct_by_rid.get(new_tgt)
                        if cur_s and new_s and cur_s.bbox and new_s.bbox:
                            if new_s.bbox.y > cur_s.bbox.y:
                                best_opt_targets[key] = rel
                else:
                    relationships.append(rel)

            relationships.extend(best_opt_targets.values())

            struct_source = meta.get("structure_source", "VLM_SUCCESS") if structures else "DETERMINISTIC_FALLBACK"
            completeness = self._compute_semantic_completeness(
                finish_reason=raw_finish_reason,
                structures=structures,
                ocr_blocks=ocr_blocks,
                image_dimensions=meta.get("image_dimensions"),
            )

            return VLMPageUnderstanding(
                page_number=page_number,
                page_purpose=page_purpose,
                document_purpose=document_purpose,
                structures=structures,
                relationships=relationships,
                raw_response=response_text[:2000],
                vlm_model=meta.get("model", self.model_name),
                image_sent=page_b64_sent,
                image_dimensions=meta.get("image_dimensions"),
                image_bytes=meta.get("image_bytes", 0),
                base64_chars=meta.get("base64_chars", 0),
                ocr_blocks_sent=len(ocr_blocks),
                prompt_chars=meta.get("prompt_chars", 0),
                vlm_attempt=True,
                structure_source=struct_source,
                vlm_provider=meta.get("provider", "gemini"),
                vlm_result="SUCCESS" if structures else "VLM_NO_STRUCTURES",
                finish_reason=raw_finish_reason,
                semantic_completeness=completeness,
                retry_count=meta.get("retry_count", 0),
                fallback_provider=meta.get("fallback_provider", "N/A"),
                structures_produced=len(structures),
                relationships_produced=len(relationships),
            )

        except Exception as e:
            print(f"[VLM] Page {page_number}: Parse error: {e}")
            return VLMPageUnderstanding(
                page_number=page_number,
                raw_response=response_text[:500],
                image_sent=page_b64_sent,
                image_dimensions=meta.get("image_dimensions"),
                image_bytes=meta.get("image_bytes", 0),
                base64_chars=meta.get("base64_chars", 0),
                ocr_blocks_sent=len(ocr_blocks),
                prompt_chars=meta.get("prompt_chars", 0),
                vlm_attempt=True,
                vlm_model=meta.get("model", self.model_name),
                structure_source="DETERMINISTIC_FALLBACK",
                vlm_provider=meta.get("provider", "gemini"),
                vlm_result="FAILED",
                finish_reason=raw_finish_reason,
                semantic_completeness="UNKNOWN",
                retry_count=meta.get("retry_count", 0),
                fallback_provider=meta.get("fallback_provider", "N/A"),
                structures_produced=0,
                relationships_produced=0,
            )

    def _compute_semantic_completeness(
        self,
        finish_reason: str,
        structures: List,
        ocr_blocks: List[Block],
        image_dimensions: Optional[Any] = None,
    ) -> str:
        """
        Real semantic completeness scorer — Fix 1.

        Does NOT blindly treat finishReason==STOP as COMPLETE.
        A VLM can stop generating having only covered part of a page.

        States:
          COMPLETE  — STOP + good coverage evidence
          PARTIAL   — MAX_TOKENS (truncated output)
          AMBIGUOUS — STOP but sparse/questionable coverage
          FAILED    — 0 structures produced
        """
        # MAX_TOKENS always means partial — output was cut off
        if finish_reason in ("MAX_TOKENS", "LENGTH", "RECITATION"):
            return "PARTIAL"

        # No structures produced → failed understanding
        if not structures:
            return "FAILED"

        # STOP + structures: evaluate actual coverage quality
        if finish_reason in ("STOP", "stop", "", "N/A"):
            total_ocr = len(ocr_blocks)
            n_structures = len(structures)

            # Check 1: Structure count relative to OCR block count
            # If VLM only identified 1-2 structures on a page with 30+ OCR blocks,
            # that is suspicious. A reasonable threshold: at least 10% coverage or
            # at least 3 structures for a substantial page.
            if total_ocr > 15 and n_structures < 2:
                print(
                    f"[VLM] SemanticCoverage: STOP but only {n_structures} structure(s) "
                    f"for {total_ocr} OCR blocks → AMBIGUOUS"
                )
                return "AMBIGUOUS"

            # Check 2: Count unique OCR block IDs referenced by VLM structures
            referenced_ids: set = set()
            for s in structures:
                for rid in getattr(s, "region_ids", []):
                    referenced_ids.add(rid)
                for rid in getattr(s, "grounded_region_ids", []):
                    referenced_ids.add(rid)

            if total_ocr > 0:
                coverage_ratio = len(referenced_ids) / total_ocr
                # A ratio below 10% on a page with >10 blocks suggests sparse VLM scan
                if total_ocr > 10 and coverage_ratio < 0.08 and n_structures < max(3, total_ocr // 10):
                    print(
                        f"[VLM] SemanticCoverage: STOP but coverage_ratio={coverage_ratio:.2f} "
                        f"({len(referenced_ids)}/{total_ocr} blocks referenced) → AMBIGUOUS"
                    )
                    return "AMBIGUOUS"

            # Check 3: Validate that structures have at least some valid bboxes or region_ids
            valid_structures = [
                s for s in structures
                if (getattr(s, "region_ids", []) or getattr(s, "grounded_region_ids", []) or getattr(s, "bbox", None))
            ]
            if len(valid_structures) == 0:
                return "AMBIGUOUS"

            # All checks passed: STOP + reasonable coverage
            return "COMPLETE"

        # Other finish reasons (SAFETY, UNKNOWN, etc.) → treat as ambiguous
        return "AMBIGUOUS"

    # ================================================================
    # LEGACY — Region Verification (backward compatibility)
    # ================================================================

    def verify_structure(
        self,
        result: DocumentUnderstandingResult,
        page_images: Optional[Dict[int, bytes]] = None,
        target_region_ids: Optional[List[str]] = None,
        force_vlm_verification: bool = False,
        **kwargs: Any,
    ) -> VisualVerificationResponse:
        """Legacy region-verification mode. Preserved for backward compatibility and tests."""
        if self.mock_response is not None:
            return self.mock_response

        if not self.is_configured(force_vlm=force_vlm_verification):
            return VisualVerificationResponse(
                status="NOT_CONFIGURED",
                is_available=False,
                model_name=self.model_name,
                vlm_hypotheses=[],
                cost_accounting=CostAccounting(
                    pages_considered=len(result.pages),
                    regions_considered=len(result.regions),
                    skipped_high_confidence_count=len(result.regions),
                ),
                error_message="VLM Vision Provider is not configured in settings.",
            )

        target_ids = set(target_region_ids) if target_region_ids else {
            r.region_id for r in result.regions if r.classification_conflict or r.confidence < 0.80
        }

        if not target_ids:
            return VisualVerificationResponse(
                status="SUCCESS",
                is_available=True,
                model_name=self.model_name,
                vlm_hypotheses=[],
                cost_accounting=CostAccounting(
                    pages_considered=len(result.pages),
                    regions_considered=len(result.regions),
                    skipped_high_confidence_count=len(result.regions),
                ),
            )

        target_regions = [r for r in result.regions if r.region_id in target_ids]
        pages_to_sent = len({r.page for r in target_regions})

        cost = CostAccounting(
            pages_considered=len(result.pages),
            pages_sent=pages_to_sent,
            regions_considered=len(result.regions),
            regions_sent=len(target_regions),
            vlm_calls=1,
            skipped_high_confidence_count=len(result.regions) - len(target_regions),
        )

        try:
            prompt = self._build_verification_prompt(target_regions, result)
            page_b64 = None
            if page_images and target_regions:
                p_num = target_regions[0].page
                if p_num in page_images:
                    page_b64 = self._encode_image(page_images[p_num])

            response_text = self._execute_vlm_call(prompt, page_b64)

            if not response_text:
                cost.failed_calls += 1
                return VisualVerificationResponse(
                    status="VLM_UNAVAILABLE",
                    model_name=self.model_name,
                    vlm_hypotheses=[],
                    cost_accounting=cost,
                    error_message="Multimodal VLM API returned empty response.",
                )

            hypotheses, verified_rels, rejected_rels = self._parse_and_validate_response(response_text, target_regions)
            cost.successful_calls += 1

            return VisualVerificationResponse(
                status="SUCCESS",
                model_name=self.model_name,
                vlm_hypotheses=hypotheses,
                verified_relationships=verified_rels,
                rejected_vlm_relationships=rejected_rels,
                cost_accounting=cost,
            )

        except Exception as e:
            cost.failed_calls += 1
            return VisualVerificationResponse(
                status="VLM_UNAVAILABLE",
                model_name=self.model_name,
                vlm_hypotheses=[],
                cost_accounting=cost,
                error_message=f"VLM verification exception: {str(e)}",
            )

    def _build_verification_prompt(self, target_regions: List[DocumentRegion], doc_result: DocumentUnderstandingResult) -> str:
        manifest_items = []
        regions_source = doc_result.regions if doc_result and doc_result.regions else target_regions
        for r in regions_source:
            manifest_items.append({
                "region_id": r.region_id,
                "page": r.page,
                "bbox": [round(r.bbox.x, 1), round(r.bbox.y, 1), round(r.bbox.width, 1), round(r.bbox.height, 1)],
                "ocr_text": r.text[:60],
                "initial_hypothesis": r.region_type,
            })

        return (
            "Analyze the attached document page image and the provided Region Manifest.\n"
            "Produce structured layout verification and relationships in valid JSON format:\n"
            "{\n"
            '  "verifications": [\n'
            '    {"region_id": "string", "proposed_type": "QUESTION"|"OPTION"|"SUBQUESTION"|"SECTION_HEADER"|"INSTRUCTION"|"HEADER"|"FOOTER"|"TABLE"|"DIAGRAM", "confidence": float, "reasoning": "string"}\n'
            "  ],\n"
            '  "relationships": [\n'
            '    {"source_region_id": "string", "target_region_id": "string", "relationship_type": "option_of"|"subquestion_of"|"section_member"|"continuation_of"|"follows", "confidence": float}\n'
            "  ]\n"
            "}\n"
            f"Region Manifest:\n{json.dumps(manifest_items, indent=2)}\n"
        )

    def _parse_and_validate_response(
        self, response_text: str, target_regions: List[DocumentRegion]
    ) -> Tuple[List[VLMHypothesis], List[RegionRelationship], List[Dict[str, Any]]]:
        """Parses and validates VLM JSON output for legacy verify_structure mode."""
        hypotheses: List[VLMHypothesis] = []
        relationships: List[RegionRelationship] = []
        rejected_relationships: List[Dict[str, Any]] = []

        try:
            valid_region_map = {r.region_id: r for r in target_regions}
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                return hypotheses, relationships, rejected_relationships

            data = json.loads(json_match.group(0))
            verifications = data.get("verifications", [])
            raw_relationships = data.get("relationships", [])

            valid_types = {
                "HEADER", "FOOTER", "METADATA", "INSTRUCTION", "SECTION_HEADER",
                "QUESTION", "SUBQUESTION", "OPTION", "TABLE", "TABLE_CELL",
                "DIAGRAM", "FIGURE", "ANSWER_SPACE", "UNKNOWN",
            }
            valid_rel_types = {
                "follows", "contains", "belongs_to", "continuation_of",
                "same_structure_as", "adjacent_to", "visually_grouped_with",
                "uncertain_relation", "option_of", "subquestion_of",
                "section_member", "associated_visual",
            }

            for item in verifications:
                reg_id = item.get("region_id")
                if not reg_id or reg_id not in valid_region_map:
                    continue
                prop_type = item.get("proposed_type", "UNKNOWN")
                if prop_type not in valid_types:
                    prop_type = "UNKNOWN"
                hypotheses.append(VLMHypothesis(
                    region_id=reg_id,
                    proposed_type=prop_type,
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=str(item.get("reasoning", "")),
                    uncertainty=float(item.get("uncertainty", 0.0)),
                ))

            # First pass: collect candidate edges
            candidate_edges = []
            for rel in raw_relationships:
                src_id = rel.get("source_region_id")
                tgt_id = rel.get("target_region_id")
                rel_type = rel.get("relationship_type", "belongs_to")
                if not src_id or not tgt_id:
                    rejected_relationships.append({"source": src_id, "target": tgt_id, "rel_type": rel_type, "reason": "Missing source or target ID"})
                    continue
                if src_id not in valid_region_map or tgt_id not in valid_region_map:
                    rejected_relationships.append({"source": src_id, "target": tgt_id, "rel_type": rel_type, "reason": "Region ID not in manifest"})
                    continue
                if src_id == tgt_id:
                    rejected_relationships.append({"source": src_id, "target": tgt_id, "rel_type": rel_type, "reason": "Self-referential link"})
                    continue
                if rel_type not in valid_rel_types:
                    rel_type = "belongs_to"
                candidate_edges.append((src_id, tgt_id, rel_type, float(rel.get("confidence", 0.9))))

            # Second pass: semantic consistency & contradiction filter
            edge_set = {(src, tgt): rtype for src, tgt, rtype, _ in candidate_edges}
            for src_id, tgt_id, rel_type, conf in candidate_edges:
                src_reg = valid_region_map[src_id]
                tgt_reg = valid_region_map[tgt_id]

                opposite_rel = edge_set.get((tgt_id, src_id))
                if opposite_rel:
                    if rel_type in ("continuation_of", "follows") and opposite_rel in ("follows", "continuation_of"):
                        if src_reg.page < tgt_reg.page or (src_reg.page == tgt_reg.page and src_reg.bbox.y < tgt_reg.bbox.y):
                            rejected_relationships.append({
                                "source": src_id, "target": tgt_id, "rel_type": rel_type,
                                "reason": f"Bidirectional loop: upper region rejected."
                            })
                            continue

                if rel_type == "follows":
                    if src_reg.page < tgt_reg.page or (src_reg.page == tgt_reg.page and src_reg.bbox.y < tgt_reg.bbox.y - 10.0):
                        rejected_relationships.append({
                            "source": src_id, "target": tgt_id, "rel_type": rel_type,
                            "reason": f"Inverted follows: source Y={src_reg.bbox.y} above target Y={tgt_reg.bbox.y}."
                        })
                        continue

                if rel_type == "continuation_of":
                    if src_reg.page < tgt_reg.page or (src_reg.page == tgt_reg.page and src_reg.bbox.y < tgt_reg.bbox.y):
                        rejected_relationships.append({
                            "source": src_id, "target": tgt_id, "rel_type": rel_type,
                            "reason": f"Inverted continuation: source above target."
                        })
                        continue
                    src_is_main_q = bool(re.match(r"^\s*(?:Q\s*\d+|\d+\.)", src_reg.text, re.IGNORECASE))
                    tgt_is_main_q = bool(re.match(r"^\s*(?:Q\s*\d+|\d+\.)", tgt_reg.text, re.IGNORECASE))
                    if src_is_main_q and tgt_is_main_q:
                        rejected_relationships.append({
                            "source": src_id, "target": tgt_id, "rel_type": rel_type,
                            "reason": f"Independent questions cannot be continuations of each other."
                        })
                        continue

                relationships.append(RegionRelationship(
                    source_region_id=src_id,
                    target_region_id=tgt_id,
                    relationship_type=rel_type,
                    confidence=conf,
                ))

        except Exception as e:
            print(f"[VLM] Legacy parse warning: {e}")

        return hypotheses, relationships, rejected_relationships
