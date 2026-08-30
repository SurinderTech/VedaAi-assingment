"""
Regression tests for Fix #2:
Making VLM Page Structure Authoritative and Enforcing Clean Semantic Boundaries.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import types
from unittest.mock import MagicMock

# Mock sklearn and sentence_transformers so DocumentUnderstandingService can be imported
for mod_name in ["sklearn", "sklearn.feature_extraction",
                 "sklearn.feature_extraction.text",
                 "sklearn.metrics", "sklearn.metrics.pairwise",
                 "sentence_transformers"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
import sklearn.feature_extraction.text as skt
if not hasattr(skt, 'TfidfVectorizer'):
    skt.TfidfVectorizer = type('TfidfVectorizer', (), {'__init__': lambda s, **kw: None})
import sklearn.metrics.pairwise as skp
if not hasattr(skp, 'cosine_similarity'):
    skp.cosine_similarity = lambda x, y: [[0.0]]

from app.models.schemas import (
    DocumentRegion, BBox, Block,
    VLMPageUnderstanding, VLMStructureItem, VLMRelationshipItem,
    DocumentPage, RegionRelationship, DocumentEvidence
)
from app.services.document_understanding_service import DocumentUnderstandingService


class TestVLMPageAuthority:

    def _make_service(self):
        return DocumentUnderstandingService.__new__(DocumentUnderstandingService)

    def _make_region(self, region_id: str, text: str, page: int = 1,
                     bbox: BBox = None, region_type: str = "UNKNOWN",
                     confidence: float = 0.5) -> DocumentRegion:
        return DocumentRegion(
            region_id=region_id,
            page=page,
            text=text,
            bbox=bbox or BBox(x=10.0, y=10.0, width=100.0, height=20.0),
            region_type=region_type,
            confidence=confidence,
            evidence=[],
            relationships=[],
            uncertainty=0.0,
            classification_conflict=False,
            conflicting_hypotheses=[],
            verification_state="UNVERIFIED",
            metadata={},
        )

    # 1. COMPLETE VLM + ungrounded OCR fragment
    def test_complete_vlm_demotes_ungrounded_ocr_fragment(self):
        svc = self._make_service()
        # Suppose deterministic parser set a numbered fragment as QUESTION
        r1 = self._make_region("r_what", "1. What", page=1, region_type="QUESTION", confidence=0.80)
        r2 = self._make_region("r_real_q", "Q2. Real Question", page=1, region_type="QUESTION", confidence=0.90)

        # VLM identifies ONLY r_real_q
        struct = VLMStructureItem(
            role="QUESTION",
            display_number="2",
            confidence=0.95,
            region_ids=["r_real_q"],
        )
        vlm_und = VLMPageUnderstanding(
            page_number=1,
            structures=[struct],
            relationships=[],
            semantic_completeness="COMPLETE",
        )

        all_regions = [r1, r2]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        # r_real_q is grounded -> QUESTION
        assert r2.region_type == "QUESTION"
        assert r2.verification_state == "VERIFIED"

        # r_what is ungrounded on COMPLETE page -> demoted to UNKNOWN
        assert r1.region_type == "UNKNOWN"
        assert r1.verification_state == "UNVERIFIED"
        assert r1.confidence <= 0.40

    # 2. COMPLETE VLM + valid QUESTION
    def test_complete_vlm_valid_question_survives(self):
        svc = self._make_service()
        r1 = self._make_region("r_q1", "12. What is photosynthesis?", page=1,
                                bbox=BBox(x=10, y=50, width=300, height=30))
        struct = VLMStructureItem(
            role="QUESTION",
            display_number="12",
            confidence=0.95,
            bbox=BBox(x=8, y=48, width=305, height=35),
        )
        vlm_und = VLMPageUnderstanding(
            page_number=1,
            structures=[struct],
            relationships=[],
            semantic_completeness="COMPLETE",
        )
        all_regions = [r1]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r1.region_type == "QUESTION"
        assert r1.verification_state == "VERIFIED"
        assert r1.confidence >= 0.90

    # 3. COMPLETE VLM + valid OPTION with complete text
    def test_complete_vlm_valid_option_survives_with_complete_text(self):
        svc = self._make_service()
        r_lbl = self._make_region("r_lbl", "(A)", page=1, bbox=BBox(x=10, y=100, width=30, height=20))
        r_txt = self._make_region("r_txt", "Ethane", page=1, bbox=BBox(x=45, y=100, width=80, height=20))

        struct = VLMStructureItem(
            role="OPTION",
            display_label="A",
            confidence=0.95,
            bbox=BBox(x=8, y=98, width=120, height=25),
        )
        vlm_und = VLMPageUnderstanding(
            page_number=1,
            structures=[struct],
            relationships=[],
            semantic_completeness="COMPLETE",
        )
        all_regions = [r_lbl, r_txt]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r_lbl.region_type == "OPTION"
        assert "(A)" in r_lbl.text and "Ethane" in r_lbl.text
        assert r_txt.region_type == "UNKNOWN"
        assert r_txt.parent_region_id == "r_lbl"

    # 4. COMPLETE VLM + unrelated OCR remains textual evidence
    def test_complete_vlm_unrelated_ocr_remains_textual_evidence(self):
        svc = self._make_service()
        r_meta = self._make_region("r_noise", "30 seconds remaining", page=1, region_type="UNKNOWN")
        r_q = self._make_region("r_q1", "1. State Newton's second law", page=1, region_type="QUESTION")

        struct = VLMStructureItem(role="QUESTION", region_ids=["r_q1"], confidence=0.95)
        vlm_und = VLMPageUnderstanding(page_number=1, structures=[struct], semantic_completeness="COMPLETE")

        all_regions = [r_meta, r_q]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r_meta.region_type == "UNKNOWN"
        assert r_meta.text == "30 seconds remaining"
        assert r_q.region_type == "QUESTION"

    # 5. PARTIAL VLM marks ungrounded as UNCERTAIN
    def test_partial_vlm_marks_ungrounded_as_uncertain(self):
        svc = self._make_service()
        r1 = self._make_region("r_q1", "1. First Question", page=1, region_type="QUESTION", confidence=0.80)
        r2 = self._make_region("r_q2", "2. Second Question", page=1, region_type="QUESTION", confidence=0.80)

        # VLM only processed r_q1 before MAX_TOKENS
        struct = VLMStructureItem(role="QUESTION", region_ids=["r_q1"], confidence=0.95)
        vlm_und = VLMPageUnderstanding(page_number=1, structures=[struct], semantic_completeness="PARTIAL")

        all_regions = [r1, r2]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r1.region_type == "QUESTION"
        assert r1.verification_state == "VERIFIED"

        # r_q2 is ungrounded on PARTIAL page -> kept as QUESTION but marked UNCERTAIN
        assert r2.region_type == "QUESTION"
        assert r2.verification_state == "UNCERTAIN"
        assert r2.uncertainty >= 0.50

    # 6. AMBIGUOUS VLM marks ungrounded as UNCERTAIN
    def test_ambiguous_vlm_marks_ungrounded_as_uncertain(self):
        svc = self._make_service()
        r1 = self._make_region("r_q1", "1. Sparse Question", page=1, region_type="QUESTION", confidence=0.80)
        r2 = self._make_region("r_q2", "2. Ungrounded Question", page=1, region_type="QUESTION", confidence=0.80)

        struct = VLMStructureItem(role="QUESTION", region_ids=["r_q1"], confidence=0.90)
        vlm_und = VLMPageUnderstanding(page_number=1, structures=[struct], semantic_completeness="AMBIGUOUS")

        all_regions = [r1, r2]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r1.verification_state == "VERIFIED"
        assert r2.verification_state == "UNCERTAIN"
        assert r2.uncertainty >= 0.50

    # 7. FAILED VLM preserves deterministic evidence as UNCERTAIN
    def test_failed_vlm_preserves_deterministic_as_uncertain(self):
        svc = self._make_service()
        r1 = self._make_region("r_q1", "1. Numbered Question", page=1, region_type="QUESTION", confidence=0.80)
        vlm_und = VLMPageUnderstanding(page_number=1, structures=[], semantic_completeness="FAILED")

        all_regions = [r1]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r1.region_type == "QUESTION"
        assert r1.verification_state == "UNCERTAIN"
        assert r1.confidence <= 0.50

    # 8. Reading order (follows exists, continuation_of does NOT automatically exist)
    def test_reading_order_follows_without_automatic_continuation(self):
        svc = self._make_service()
        r1 = self._make_region("r1", "1. Question text", page=1, bbox=BBox(x=10, y=50, width=200, height=20))
        r2 = self._make_region("r2", "Next unrelated text block", page=1, bbox=BBox(x=10, y=75, width=200, height=20))

        rels = svc._extract_intra_page_relationships([r1, r2])
        rel_types = [r.relationship_type for r in rels]

        # follows should exist for reading order
        assert "follows" in rel_types
        # continuation_of must NOT be created from simple adjacency
        assert "continuation_of" not in rel_types

    # 9. Real continuation supported by VLM evidence
    def test_real_vlm_continuation_is_preserved_in_graph(self):
        svc = self._make_service()
        r_head = self._make_region("q1_head", "1. Explain the working principle", page=1, region_type="QUESTION")
        r_cont = self._make_region("q1_cont", "of a synchronous motor in detail.", page=1, region_type="UNKNOWN")

        vlm_rel = RegionRelationship(
            source_region_id="q1_cont",
            target_region_id="q1_head",
            relationship_type="continuation_of",
            confidence=0.90,
            evidence=[DocumentEvidence(
                signal_type="visual_vlm_verification",
                description="VLM grounded continuation",
                weight=0.9,
                score=0.90,
            )],
        )

        graph = svc._build_structure_graph(
            all_regions=[r_head, r_cont],
            all_relationships=[vlm_rel],
            document_purpose="QUESTION_PAPER",
            page_roles={1: "QUESTION_PAPER"},
        )

        cont_edges = [e for e in graph.edges if e.relationship == "continuation_of"]
        assert len(cont_edges) == 1
        assert cont_edges[0].source_id == "q1_cont"
        assert cont_edges[0].target_id == "q1_head"

    # 10. Option Grounding with multiple OCR blocks
    def test_option_grounding_merges_multiple_ocr_blocks_into_one_option(self):
        svc = self._make_service()
        r_a = self._make_region("opt_a_lbl", "(A)", page=1, bbox=BBox(x=10, y=100, width=25, height=18))
        r_b = self._make_region("opt_a_val", "Ethane gas", page=1, bbox=BBox(x=40, y=100, width=90, height=18))

        struct = VLMStructureItem(
            role="OPTION",
            display_label="A",
            confidence=0.92,
            bbox=BBox(x=8, y=95, width=130, height=28),
        )
        vlm_und = VLMPageUnderstanding(
            page_number=1,
            structures=[struct],
            relationships=[],
            semantic_completeness="COMPLETE",
        )

        all_regions = [r_a, r_b]
        svc._apply_vlm_page_understandings([vlm_und], all_regions, [], {})

        assert r_a.region_type == "OPTION"
        assert r_a.text == "(A) Ethane gas"
        assert r_b.region_type == "UNKNOWN"
        assert r_b.parent_region_id == "opt_a_lbl"
