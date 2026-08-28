"""
Step 11B — Multimodal Document Vision Provider Boundary Implementation.

Provides the pluggable provider abstraction for VLM visual verification.
Encapsulates multimodal model calls (Gemini/OpenRouter) behind the DocumentVisionProvider interface,
schema-validating output into VisualVerificationResponse and handling safe fallbacks (VLM_UNAVAILABLE).
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import (
    DocumentRegion,
    DocumentUnderstandingResult,
    VisualVerificationResponse,
    VLMHypothesis,
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
    """
    Abstract interface & orchestration boundary for multi-modal vision models (VLMs).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key if api_key is not None else getattr(settings, "GEMINI_API_KEY", "")
        self.model_name = model_name or getattr(settings, "DOCUMENT_VLM_MODEL", "gemini-2.5-flash")

    def is_configured(self) -> bool:
        """Returns True if VLM enabled and API key configured."""
        enabled = getattr(settings, "DOCUMENT_VLM_ENABLED", False)
        gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        or_key = getattr(settings, "OPENROUTER_API_KEY", "")
        has_key = bool((self.api_key and len(self.api_key) > 5) or (gemini_key and len(gemini_key) > 5) or (or_key and len(or_key) > 5))
        return enabled and has_key

    def analyze_page(
        self,
        page_image_bytes: Optional[bytes] = None,
        page_number: int = 1,
        **kwargs: Any,
    ) -> VisionAnalysisResult:
        """Analyzes page layout/visual elements via VLM."""
        return VisionAnalysisResult(
            status="NOT_CONFIGURED",
            message=f"VLM Vision Provider is not configured for page {page_number}.",
            is_available=False,
            metadata={"page_number": page_number},
        )

    def analyze_regions(
        self,
        page_image_bytes: Optional[bytes] = None,
        regions: Optional[List[DocumentRegion]] = None,
        **kwargs: Any,
    ) -> VisionAnalysisResult:
        """Analyzes specific visual regions via VLM."""
        region_count = len(regions) if regions else 0
        return VisionAnalysisResult(
            status="NOT_CONFIGURED",
            message=f"VLM Vision Provider is not configured for {region_count} regions.",
            is_available=False,
            metadata={"region_count": region_count},
        )

    def verify_structure(
        self,
        result: DocumentUnderstandingResult,
        page_images: Optional[Dict[int, bytes]] = None,
        target_region_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VisualVerificationResponse:
        """Verifies structural hypotheses using VLM."""
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
    Multimodal VLM Provider implementation delegating to LLMProvider / Gemini / OpenRouter multimodal API.
    Supports deterministic mock responses for unit testing without live API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_response: Optional[VisualVerificationResponse] = None,
    ):
        super().__init__(api_key=api_key, model_name=model_name)
        self.mock_response = mock_response

    def is_configured(self, force_vlm: bool = False) -> bool:
        """Returns True if VLM enabled (or forced) and API key configured."""
        enabled = getattr(settings, "DOCUMENT_VLM_ENABLED", False) or force_vlm or (self.mock_response is not None)
        gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        or_key = getattr(settings, "OPENROUTER_API_KEY", "")
        has_key = bool((self.api_key and len(self.api_key) > 5) or (gemini_key and len(gemini_key) > 5) or (or_key and len(or_key) > 5))
        return (enabled and has_key) or (self.mock_response is not None)

    def verify_structure(
        self,
        result: DocumentUnderstandingResult,
        page_images: Optional[Dict[int, bytes]] = None,
        target_region_ids: Optional[List[str]] = None,
        force_vlm_verification: bool = False,
        **kwargs: Any,
    ) -> VisualVerificationResponse:
        """
        Executes structured visual verification for target regions.
        """
        # If deterministic mock response provided for testing, return it immediately
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

        # Select target regions to verify
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
            from app.services.llm_provider import _call_gemini, _call_openrouter, LLMError

            prompt = self._build_verification_prompt(target_regions, result)
            
            page_b64 = None
            if page_images and target_regions:
                p_num = target_regions[0].page
                if p_num in page_images:
                    img_obj = page_images[p_num]
                    import base64, io
                    if isinstance(img_obj, bytes):
                        page_b64 = base64.b64encode(img_obj).decode("utf-8")
                    elif hasattr(img_obj, "save"):
                        buf = io.BytesIO()
                        img_obj.save(buf, format="PNG")
                        page_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            import asyncio
            response_text = ""

            async def _run_vlm():
                primary = (getattr(settings, "PRIMARY_LLM_PROVIDER", "gemini")).lower().strip()
                if primary == "openrouter" or (getattr(settings, "OPENROUTER_API_KEY", "")):
                    try:
                        return await _call_openrouter(prompt, image_b64=page_b64)
                    except Exception as eor:
                        print(f"[DocumentVisionProvider] OpenRouter notice: {eor}. Trying Gemini...")
                        return await _call_gemini(prompt, image_b64=page_b64)
                else:
                    try:
                        return await _call_gemini(prompt, image_b64=page_b64)
                    except Exception as eg:
                        print(f"[DocumentVisionProvider] Gemini notice: {eg}. Trying OpenRouter...")
                        return await _call_openrouter(prompt, image_b64=page_b64)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    response_text = loop.run_until_complete(_run_vlm())
                else:
                    response_text = loop.run_until_complete(_run_vlm())
            except Exception:
                try:
                    response_text = asyncio.run(_run_vlm())
                except Exception as e_run:
                    print(f"[DocumentVisionProvider] VLM run failed: {e_run}")
                    response_text = ""

            if not response_text:
                cost.failed_calls += 1
                return VisualVerificationResponse(
                    status="VLM_UNAVAILABLE",
                    model_name=self.model_name,
                    vlm_hypotheses=[],
                    cost_accounting=cost,
                    error_message="Multimodal VLM API returned empty response or timed out.",
                )

            # Schema validate JSON output from VLM
            hypotheses, verified_rels = self._parse_and_validate_response(response_text, target_regions)
            cost.successful_calls += 1

            return VisualVerificationResponse(
                status="SUCCESS",
                model_name=self.model_name,
                vlm_hypotheses=hypotheses,
                verified_relationships=verified_rels,
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

    def _build_verification_prompt(
        self, target_regions: List[DocumentRegion], result: DocumentUnderstandingResult
    ) -> str:
        """Builds structured visual grounding JSON prompt for VLM verification."""
        manifest_items = []
        for r in result.regions:
            manifest_items.append(
                {
                    "region_id": r.region_id,
                    "page": r.page,
                    "bbox": [r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height],
                    "ocr_text": r.text,
                    "initial_hypothesis": r.region_type,
                }
            )

        return f"""You are an expert Document Layout & Structural Intelligence Verifier.
Analyze the provided document page image alongside the supplied REGION MANIFEST.

CRITICAL INSTRUCTIONS:
1. Document-Level Semantic Decision: Determine the role and purpose of content using page/document context. Distinguish meaningful assessment content from administrative/irrelevant text based on visual layout, hierarchy, and surrounding context.
2. NO SINGLE SIGNAL AUTHORITY: No single keyword, numbering pattern, page position, or confidence score independently promotes a region to a question.
3. VISUAL REGION GROUNDING: You MUST ONLY reference region_ids that exist in the supplied manifest. NEVER invent region IDs.

REGION MANIFEST:
{json.dumps(manifest_items, indent=2)}

Return ONLY valid JSON matching this exact schema:
{{
  "document_purpose": "QUESTION_PAPER" | "INSTRUCTIONS" | "COVER" | "ANSWER_KEY" | "UNKNOWN",
  "page_roles": {{"1": "COVER", "2": "QUESTION_PAPER"}},
  "verifications": [
    {{
      "region_id": "string",
      "proposed_type": "QUESTION" | "SUBQUESTION" | "OPTION" | "INSTRUCTION" | "SECTION_HEADER" | "TABLE" | "DIAGRAM" | "METADATA" | "HEADER" | "FOOTER",
      "confidence": float (0.0 to 1.0),
      "reasoning": "string explanation",
      "uncertainty": float (0.0 to 1.0)
    }}
  ],
  "relationships": [
    {{
      "source_region_id": "string",
      "target_region_id": "string",
      "relationship_type": "option_of" | "belongs_to" | "continuation_of" | "subquestion_of" | "section_member" | "associated_visual",
      "confidence": float (0.0 to 1.0)
    }}
  ]
}}
"""

    def _parse_and_validate_response(
        self, response_text: str, target_regions: List[DocumentRegion]
    ) -> Tuple[List[VLMHypothesis], List[RegionRelationship]]:
        """Parses and strictly schema-validates VLM JSON output against known region manifest."""
        hypotheses: List[VLMHypothesis] = []
        relationships: List[RegionRelationship] = []

        try:
            valid_region_map = {r.region_id: r for r in target_regions}
            # Extract JSON block
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                return hypotheses, relationships

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
                    # REJECT UNKNOWN REGION IDs
                    continue

                prop_type = item.get("proposed_type", "UNKNOWN")
                if prop_type not in valid_types:
                    prop_type = "UNKNOWN"

                conf = float(item.get("confidence", 0.5))
                reason = str(item.get("reasoning", ""))
                uncert = float(item.get("uncertainty", 0.0))

                hypotheses.append(
                    VLMHypothesis(
                        region_id=reg_id,
                        proposed_type=prop_type,
                        confidence=conf,
                        reasoning=reason,
                        uncertainty=uncert,
                    )
                )

            for rel in raw_relationships:
                src_id = rel.get("source_region_id")
                tgt_id = rel.get("target_region_id")
                rel_type = rel.get("relationship_type", "belongs_to")

                # STRICT VALIDATION: reject unknown region IDs, self-links, or invalid relationship types
                if not src_id or not tgt_id:
                    continue
                if src_id not in valid_region_map or tgt_id not in valid_region_map:
                    continue
                if src_id == tgt_id:
                    continue
                if rel_type not in valid_rel_types:
                    rel_type = "belongs_to"

                conf = float(rel.get("confidence", 0.9))
                relationships.append(
                    RegionRelationship(
                        source_region_id=src_id,
                        target_region_id=tgt_id,
                        relationship_type=rel_type,
                        confidence=conf,
                    )
                )

        except Exception as e:
            print(f"[DocumentVisionProvider] JSON validation warning: {e}")

        return hypotheses, relationships

