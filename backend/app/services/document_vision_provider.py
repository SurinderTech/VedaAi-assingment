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

        return understanding

    def _build_page_understanding_prompt(
        self,
        ocr_blocks: List[Block],
        page_number: int,
        total_pages: int,
        page_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Builds the page-level document understanding prompt.

        The VLM is asked to UNDERSTAND the document, not verify pre-existing hypotheses.
        Full OCR text is provided — NOT truncated.
        """
        # Build complete OCR evidence with IDs and bounding boxes
        ocr_evidence_lines = []
        for b in ocr_blocks:
            ocr_evidence_lines.append(
                f'  {{"id": "{b.id}", "text": {json.dumps(b.text)}, '
                f'"bbox": [{round(b.bbox.x,1)}, {round(b.bbox.y,1)}, {round(b.bbox.width,1)}, {round(b.bbox.height,1)}], '
                f'"source": "{b.source or "ocr"}"}}'
            )

        ocr_evidence = "[\n" + ",\n".join(ocr_evidence_lines) + "\n]"

        context_info = ""
        if page_context:
            if page_context.get("prev_page_summary"):
                context_info += f"\nPrevious page context: {page_context['prev_page_summary']}"
            if page_context.get("next_page_summary"):
                context_info += f"\nNext page context: {page_context['next_page_summary']}"

        return f"""You are analyzing page {page_number} of {total_pages} from an academic document.

TASK: Examine the attached page image AND the OCR text evidence below. Determine what this page contains and how its content is structured.
{context_info}
IMPORTANT RULES:
- A numbered item is NOT automatically a question. Consider the document context.
  "1. SECTION-A is COMPULSORY consisting of TEN questions..." is INSTRUCTION / SECTION_HEADER content.
  "2. SECTION-B contains FIVE questions..." is INSTRUCTION / SECTION_HEADER content.
  "3. SECTION-C contains THREE questions..." is INSTRUCTION / SECTION_HEADER content.
  "General Instructions: 1. Answer all questions" — these are INSTRUCTIONS, not questions.
- Reference ONLY the supplied OCR region IDs. Do NOT invent new IDs.
- Do NOT generate replacement text. The OCR text is the authoritative source.
- Keep reasoning string concise (max 6 words per entry) to fit response limits.
- Group contiguous OCR text blocks for the same question into a single structure entry region_ids array.

OCR Evidence (ordered by reading position on page {page_number}):
{ocr_evidence}

Respond in valid JSON:
{{
  "page_purpose": "QUESTION_PAGE" | "COVER" | "INSTRUCTIONS" | "CONTINUATION" | "MIXED" | "ADMINISTRATIVE",
  "document_purpose": "EXAMINATION_PAPER" | "ASSIGNMENT" | "INSTRUCTIONS" | "UNKNOWN",
  "structures": [
    {{
      "region_ids": ["id1", "id2"],
      "role": "QUESTION" | "OPTION" | "SUBQUESTION" | "SECTION_HEADER" | "INSTRUCTION" | "METADATA" | "HEADER" | "FOOTER" | "ADMINISTRATIVE" | "TABLE" | "DIAGRAM" | "CONTINUATION" | "UNKNOWN",
      "display_number": "1",
      "display_label": "Section A",
      "reasoning": "Why this content has this role in the document context",
      "confidence": 0.95
    }}
  ],
  "relationships": [
    {{
      "source_ids": ["id3"],
      "target_ids": ["id1"],
      "type": "option_of" | "subquestion_of" | "section_member" | "continuation_of" | "belongs_to" | "follows",
      "confidence": 0.9
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
        """Parses VLM JSON response into structured VLMPageUnderstanding, validating region IDs."""
        valid_ids = {b.id for b in ocr_blocks}
        meta = vlm_meta or {}

        try:
            from app.services.llm_provider import extract_json_payload
            data = extract_json_payload(response_text)
            if not isinstance(data, dict):
                data = {}

            page_purpose = data.get("page_purpose", "UNKNOWN")
            document_purpose = data.get("document_purpose", "UNKNOWN")

            # Parse structures — validate region IDs against real OCR blocks
            structures: List[VLMStructureItem] = []
            for item in data.get("structures", []):
                raw_ids = item.get("region_ids", [])
                validated_ids = [rid for rid in raw_ids if rid in valid_ids]
                if not validated_ids:
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
                    role=normalized_role,
                    display_number=item.get("display_number"),
                    display_label=item.get("display_label"),
                    reasoning=str(item.get("reasoning", "")),
                    confidence=float(item.get("confidence", 0.5)),
                ))

            # Parse relationships — validate all referenced IDs
            relationships: List[VLMRelationshipItem] = []
            for rel in data.get("relationships", []):
                src_ids = [rid for rid in rel.get("source_ids", []) if rid in valid_ids]
                tgt_ids = [rid for rid in rel.get("target_ids", []) if rid in valid_ids]
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

                relationships.append(VLMRelationshipItem(
                    source_ids=src_ids,
                    target_ids=tgt_ids,
                    relationship_type=rel_type,
                    confidence=float(rel.get("confidence", 0.5)),
                ))

            struct_source = meta.get("structure_source", "VLM_SUCCESS") if structures else "DETERMINISTIC_FALLBACK"

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
                finish_reason=meta.get("finish_reason", "STOP"),
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
                finish_reason=meta.get("finish_reason", "N/A"),
                retry_count=meta.get("retry_count", 0),
                fallback_provider=meta.get("fallback_provider", "N/A"),
                structures_produced=0,
                relationships_produced=0,
            )


            # Parse structures — validate region IDs against real OCR blocks
            structures: List[VLMStructureItem] = []
            for item in data.get("structures", []):
                raw_ids = item.get("region_ids", [])
                # Only keep IDs that exist in the real OCR block set
                validated_ids = [rid for rid in raw_ids if rid in valid_ids]
                if not validated_ids:
                    # VLM referenced unknown IDs — skip this structure entirely
                    continue

                role = item.get("role", "UNKNOWN")
                # Normalize role to valid DocumentRegionType
                role_map = {
                    "ADMINISTRATIVE": "INSTRUCTION",
                    "CONTINUATION": "UNKNOWN",  # Handled via relationships
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
                    role=normalized_role,
                    display_number=item.get("display_number"),
                    display_label=item.get("display_label"),
                    reasoning=str(item.get("reasoning", "")),
                    confidence=float(item.get("confidence", 0.5)),
                ))

            # Parse relationships — validate all referenced IDs
            relationships: List[VLMRelationshipItem] = []
            for rel in data.get("relationships", []):
                src_ids = [rid for rid in rel.get("source_ids", []) if rid in valid_ids]
                tgt_ids = [rid for rid in rel.get("target_ids", []) if rid in valid_ids]
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

                relationships.append(VLMRelationshipItem(
                    source_ids=src_ids,
                    target_ids=tgt_ids,
                    relationship_type=rel_type,
                    confidence=float(rel.get("confidence", 0.5)),
                ))

            return VLMPageUnderstanding(
                page_number=page_number,
                page_purpose=page_purpose,
                document_purpose=document_purpose,
                structures=structures,
                relationships=relationships,
                raw_response=response_text[:2000],
                vlm_model=self.model_name,
                image_sent=page_b64_sent,
                ocr_blocks_sent=len(ocr_blocks),
            )

        except Exception as e:
            print(f"[VLM] Page {page_number}: Parse error: {e}")
            return VLMPageUnderstanding(
                page_number=page_number,
                raw_response=response_text[:500],
                image_sent=page_b64_sent,
                ocr_blocks_sent=len(ocr_blocks),
                vlm_model=self.model_name,
            )

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
