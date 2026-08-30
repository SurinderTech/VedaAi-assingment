"""
Unit tests for VedaAI 5-Fix Global Hardening Pass.

FIX 1 — Semantic completeness scoring
FIX 2 — Duplicate/ghost question deduplication
FIX 3 — Authoritative VLM boundary enforcement
FIX 4 — Cross-page continuation edge semantics
FIX 5 — API/frontend structural preservation

Run with:
    cd backend
    py -3 -m pytest tests/test_vlm_hardening.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock


# ============================================================
# FIX 1 — Semantic Completeness Scoring
# ============================================================

class TestSemanticCompleteness:

    def _make_provider(self):
        from app.services.document_vision_provider import MultimodalDocumentVisionProvider
        provider = MultimodalDocumentVisionProvider.__new__(MultimodalDocumentVisionProvider)
        provider.model_name = "test-model"
        return provider

    def _make_struct(self, region_ids=None, bbox=None):
        s = MagicMock()
        s.region_ids = region_ids or []
        s.grounded_region_ids = []
        s.bbox = bbox
        return s

    def _make_blocks(self, n: int):
        from app.models.schemas import Block, BBox
        return [
            Block(id=f"blk_{i}", text=f"text {i}", confidence=0.9,
                  bbox=BBox(x=0, y=i*20, width=100, height=18), page=1)
            for i in range(n)
        ]

    def test_max_tokens_always_partial(self):
        provider = self._make_provider()
        result = provider._compute_semantic_completeness(
            finish_reason="MAX_TOKENS",
            structures=[self._make_struct(["blk_0"])],
            ocr_blocks=self._make_blocks(5),
        )
        assert result == "PARTIAL"

    def test_stop_zero_structures_is_failed(self):
        provider = self._make_provider()
        result = provider._compute_semantic_completeness(
            finish_reason="STOP",
            structures=[],
            ocr_blocks=self._make_blocks(20),
        )
        assert result == "FAILED"

    def test_stop_good_coverage_is_complete(self):
        provider = self._make_provider()
        blocks = self._make_blocks(10)
        structs = [self._make_struct(region_ids=[f"blk_{i}" for i in range(8)])]
        result = provider._compute_semantic_completeness(
            finish_reason="STOP",
            structures=structs,
            ocr_blocks=blocks,
        )
        assert result == "COMPLETE"

    def test_stop_sparse_coverage_is_ambiguous(self):
        provider = self._make_provider()
        blocks = self._make_blocks(20)
        structs = [self._make_struct(region_ids=["blk_0"])]
        result = provider._compute_semantic_completeness(
            finish_reason="STOP",
            structures=structs,
            ocr_blocks=blocks,
        )
        assert result == "AMBIGUOUS"

    def test_safety_finish_reason_is_ambiguous(self):
        provider = self._make_provider()
        result = provider._compute_semantic_completeness(
            finish_reason="SAFETY",
            structures=[self._make_struct(["blk_0"])],
            ocr_blocks=self._make_blocks(5),
        )
        assert result == "AMBIGUOUS"


# ============================================================
# FIX 4 — Cross-Page Continuation Edge Semantics
# ============================================================

class TestCrossPageContinuation:

    def _make_region(self, region_id, region_type, text, page=1):
        from app.models.schemas import DocumentRegion, BBox
        return DocumentRegion(
            region_id=region_id, page=page, text=text,
            bbox=BBox(x=0, y=10, width=200, height=20),
            region_type=region_type,
        )

    def _get_service(self):
        """Import DocumentUnderstandingService with mocked heavy deps."""
        import sys
        import types
        # Mock sklearn and sentence_transformers so the import chain works
        for mod_name in ["sklearn", "sklearn.feature_extraction",
                         "sklearn.feature_extraction.text",
                         "sklearn.metrics", "sklearn.metrics.pairwise",
                         "sentence_transformers"]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
        # Provide a stub TfidfVectorizer
        import sklearn.feature_extraction.text as skt
        if not hasattr(skt, 'TfidfVectorizer'):
            skt.TfidfVectorizer = type('TfidfVectorizer', (), {'__init__': lambda s, **kw: None})
        import sklearn.metrics.pairwise as skp
        if not hasattr(skp, 'cosine_similarity'):
            skp.cosine_similarity = lambda x, y: [[0.0]]

        from app.services.document_understanding_service import DocumentUnderstandingService
        return DocumentUnderstandingService()

    def test_no_continuation_when_next_page_starts_with_question(self):
        from app.models.schemas import DocumentPage
        svc = self._get_service()
        last_q_p1 = self._make_region("q5", "QUESTION", "End of Q5 body", page=1)
        first_q_p2 = self._make_region("q6", "QUESTION", "Q6. What is osmosis?", page=2)
        page1 = DocumentPage(page_number=1, regions=[last_q_p1])
        page2 = DocumentPage(page_number=2, regions=[first_q_p2])

        rels = svc._extract_cross_page_relationships([page1, page2])
        cont_rels = [r for r in rels if r.relationship_type == "continuation_of"]
        assert len(cont_rels) == 0, \
            f"Should NOT create cross-page continuation when next page starts with QUESTION, got {len(cont_rels)}"

    def test_continuation_allowed_for_unknown_first_region(self):
        from app.models.schemas import DocumentPage
        svc = self._get_service()
        last_q_p1 = self._make_region("q5", "QUESTION", "This question continues", page=1)
        continuation_p2 = self._make_region("cont1", "UNKNOWN", "on the next page.", page=2)
        page1 = DocumentPage(page_number=1, regions=[last_q_p1])
        page2 = DocumentPage(page_number=2, regions=[continuation_p2])

        rels = svc._extract_cross_page_relationships([page1, page2])
        cont_rels = [r for r in rels if r.relationship_type == "continuation_of"]
        assert len(cont_rels) == 1, \
            f"Should create cross-page continuation for UNKNOWN first region, got {len(cont_rels)}"


# ============================================================
# FIX 5 — API/Frontend Structural Preservation
# ============================================================

class TestAPIStructuralPreservation:

    def test_options_list_preserved(self):
        from app.models.schemas import MappedAnswer, QuestionResult, Grading
        from app.services.assessment_result_service import build_structured_assessment_result

        q_result = QuestionResult(
            id="q2", number="2",
            text="Q2. Which element is used in photosynthesis?",
            page=1,
            answer=MappedAnswer(status="unanswered", confidence=0.0, regions=[]),
            grading=Grading(score=0.0, max_score=1.0),
            options=["A. Carbon Dioxide", "B. Nitrogen", "C. Hydrogen", "D. Oxygen"],
        )
        result = build_structured_assessment_result("test_id_2", [q_result], [])
        sqr = result.question_results[0]
        assert sqr.options == ["A. Carbon Dioxide", "B. Nitrogen", "C. Hydrogen", "D. Oxygen"]

    def test_question_type_propagated(self):
        from app.models.schemas import MappedAnswer, QuestionResult, Grading
        from app.services.assessment_result_service import build_structured_assessment_result

        q_result = QuestionResult(
            id="q1", number="1",
            text="Q1. What is photosynthesis?",
            page=2,
            answer=MappedAnswer(status="matched", answer_id="ans_1", text="A", confidence=0.9, regions=[]),
            grading=Grading(score=2.0, max_score=2.0),
            options=["A. Sugar", "B. ATP", "C. Glucose", "D. Water"],
        )
        q_result.__dict__['question_type'] = "MCQ"
        q_result.__dict__['source_region_ids'] = ["reg_1"]
        q_result.__dict__['extraction_confidence'] = 0.92
        q_result.__dict__['verification_state'] = "VERIFIED"
        q_result.__dict__['parent_question_id'] = None

        result = build_structured_assessment_result("test_id", [q_result], [])
        sqr = result.question_results[0]
        assert sqr.question_type == "MCQ"
        assert sqr.page_number == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
