"""
Step 11B — Provider-Agnostic Evidence Fusion Service.

Combines deterministic, layout, semantic, and VLM visual evidence into
calibrated multi-hypothesis structure classifications and explicit verification states:
VERIFIED, CONFLICTED, UNCERTAIN, or UNVERIFIED.

Strictly provider-agnostic with zero vendor-specific code.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any, Set

from app.models.schemas import (
    DocumentRegion,
    DocumentUnderstandingResult,
    StructureHypothesis,
    DocumentEvidence,
    RegionRelationship,
    DocumentRegionType,
    VerificationState,
    VisualVerificationResponse,
    VLMHypothesis,
)


class EvidenceFusionService:
    """
    Fuses evidence across Parser, Layout, Semantic, and VLM hypothesis sources.
    """

    # Reliability weights for hypothesis sources
    SOURCE_WEIGHTS: Dict[str, float] = {
        "parser": 0.85,
        "layout_analyzer": 0.80,
        "semantic_analyzer": 0.75,
        "vlm": 0.90,
    }

    def fuse_document_evidence(
        self,
        understanding_result: DocumentUnderstandingResult,
        vlm_response: Optional[VisualVerificationResponse] = None,
    ) -> DocumentUnderstandingResult:
        """
        Fuses initial Step 11A hypotheses with optional VLM hypotheses.
        Updates region classifications, verification states, and conflict listings.
        """
        if not understanding_result.regions:
            return understanding_result

        # Map VLM hypotheses by region_id if available
        vlm_hyp_map: Dict[str, VLMHypothesis] = {}
        if vlm_response and vlm_response.status == "SUCCESS":
            for v_hyp in vlm_response.vlm_hypotheses:
                vlm_hyp_map[v_hyp.region_id] = v_hyp

        verified_count = 0
        conflicted_count = 0
        uncertain_count = 0
        unverified_count = 0

        updated_regions: List[DocumentRegion] = []
        global_conflicts: List[Dict[str, Any]] = list(understanding_result.conflicts)

        for reg in understanding_result.regions:
            vlm_hyp = vlm_hyp_map.get(reg.region_id)

            # If VLM supplied hypothesis, attach as separate hypothesis source
            if vlm_hyp:
                vlm_evidence = [
                    DocumentEvidence(
                        signal_type="visual_vlm_verification",
                        description=f"VLM visual verification: {vlm_hyp.reasoning}",
                        weight=0.5,
                        score=vlm_hyp.confidence,
                    )
                ]
                vlm_struct_hyp = StructureHypothesis(
                    region_id=reg.region_id,
                    hypothesized_type=vlm_hyp.proposed_type,
                    confidence=vlm_hyp.confidence,
                    source="vlm",
                    evidence=vlm_evidence,
                )
                reg.vlm_hypothesis = vlm_struct_hyp

                # Preserve all existing hypotheses and append VLM hypothesis
                existing_sources = {h.source for h in reg.conflicting_hypotheses}
                if "vlm" not in existing_sources:
                    reg.conflicting_hypotheses.append(vlm_struct_hyp)
                reg.evidence.extend(vlm_evidence)

            # Fuse hypotheses for this region
            v_state, fused_type, fused_conf, has_conflict = self._fuse_region_hypotheses(
                reg, vlm_hyp_provided=(vlm_hyp is not None)
            )

            reg.verification_state = v_state
            reg.region_type = fused_type
            reg.confidence = fused_conf
            reg.classification_conflict = has_conflict

            if v_state == "VERIFIED":
                verified_count += 1
            elif v_state == "CONFLICTED":
                conflicted_count += 1
                global_conflicts.append(
                    {
                        "region_id": reg.region_id,
                        "page": reg.page,
                        "text": reg.text[:60],
                        "verification_state": "CONFLICTED",
                        "competing_sources": [
                            {"source": h.source, "type": h.hypothesized_type, "confidence": h.confidence}
                            for h in reg.conflicting_hypotheses
                        ],
                    }
                )
            elif v_state == "UNCERTAIN":
                uncertain_count += 1
            else:
                unverified_count += 1

            updated_regions.append(reg)

        # Merge verified relationships from VLM if available
        all_relationships = list(understanding_result.relationships)
        if vlm_response and vlm_response.verified_relationships:
            existing_rel_keys = {
                (r.source_region_id, r.target_region_id, r.relationship_type)
                for r in all_relationships
            }
            for v_rel in vlm_response.verified_relationships:
                rel_key = (v_rel.source_region_id, v_rel.target_region_id, v_rel.relationship_type)
                if rel_key not in existing_rel_keys:
                    all_relationships.append(v_rel)
                    existing_rel_keys.add(rel_key)

        understanding_result.regions = updated_regions
        understanding_result.relationships = all_relationships
        understanding_result.conflicts = global_conflicts
        understanding_result.verification_summary = {
            "total_regions": len(updated_regions),
            "verified_count": verified_count,
            "conflicted_count": conflicted_count,
            "uncertain_count": uncertain_count,
            "unverified_count": unverified_count,
            "vlm_response_status": vlm_response.status if vlm_response else "UNVERIFIED",
        }

        return understanding_result

    def _fuse_region_hypotheses(
        self, reg: DocumentRegion, vlm_hyp_provided: bool
    ) -> Tuple[VerificationState, DocumentRegionType, float, bool]:
        """
        Calculates fused score across hypothesis sources and assigns verification state.
        Returns (verification_state, fused_type, fused_confidence, classification_conflict).
        """
        hypotheses = reg.conflicting_hypotheses
        if not hypotheses:
            return ("UNVERIFIED" if not vlm_hyp_provided else "UNCERTAIN", reg.region_type, reg.confidence, False)

        # Group weighted confidence score by hypothesized_type
        type_scores: Dict[DocumentRegionType, float] = {}
        type_weight_sums: Dict[DocumentRegionType, float] = {}

        for h in hypotheses:
            w = self.SOURCE_WEIGHTS.get(h.source, 0.70)
            type_scores[h.hypothesized_type] = type_scores.get(h.hypothesized_type, 0.0) + (h.confidence * w)
            type_weight_sums[h.hypothesized_type] = type_weight_sums.get(h.hypothesized_type, 0.0) + w

        normalized_scores: Dict[DocumentRegionType, float] = {}
        for t_type, total_score in type_scores.items():
            normalized_scores[t_type] = total_score / max(1.0, type_weight_sums[t_type])

        sorted_types = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        top_type, top_score = sorted_types[0]

        # Determine if strong multi-source agreement or conflict exists
        distinct_types = len(sorted_types)
        has_conflict = False

        if distinct_types > 1:
            second_type, second_score = sorted_types[1]
            # Significant disagreement between two high-scoring sources
            if top_score >= 0.70 and second_score >= 0.70:
                has_conflict = True
                return ("CONFLICTED", top_type, top_score * 0.85, True)

        # Determine verification state
        if vlm_hyp_provided:
            vlm_h = next((h for h in hypotheses if h.source == "vlm"), None)
            if vlm_h and vlm_h.hypothesized_type == top_type and top_score >= 0.75:
                return ("VERIFIED", top_type, max(top_score, vlm_h.confidence), has_conflict)
            elif top_score >= 0.80 and not has_conflict:
                return ("VERIFIED", top_type, top_score, False)
            elif has_conflict:
                return ("CONFLICTED", top_type, top_score, True)
            else:
                return ("UNCERTAIN", top_type, top_score, False)

        # When VLM was not provided/called (e.g. skipped high confidence or unconfigured)
        if top_score >= 0.85 and not has_conflict:
            return ("VERIFIED", top_type, top_score, False)
        elif has_conflict:
            return ("CONFLICTED", top_type, top_score, True)
        elif top_score < 0.60:
            return ("UNCERTAIN", top_type, top_score, False)

        return ("UNVERIFIED", top_type, top_score, has_conflict)
