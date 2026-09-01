import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.schemas import (
    AnswerRegion,
    BBox,
    Block,
    DocumentRegion,
    Region,
    StructuredAnswerSheet,
    VLMPageUnderstanding,
    VLMStructureItem,
)
from app.services.answer_extractor import _augment_with_vlm_evidence
from app.services.document_understanding_service import DocumentUnderstandingService


class TestPhase1VlmAnswerEvidence:
    def _make_doc_region(self, region_id, text, page=1, bbox=None, role="UNKNOWN", confidence=0.9, metadata=None):
        return DocumentRegion(
            region_id=region_id,
            page=page,
            text=text,
            bbox=bbox or BBox(x=10, y=10, width=100, height=20),
            region_type=role,
            confidence=confidence,
            evidence=[],
            relationships=[],
            uncertainty=0.0,
            classification_conflict=False,
            conflicting_hypotheses=[],
            verification_state="UNVERIFIED",
            metadata=metadata or {},
        )

    def test_01_vlm_answer_relationship_survives_in_answer_region(self):
        structured = StructuredAnswerSheet(
            num_pages=1,
            page_analyses={},
            answer_regions=[],
            unanchored_regions=[],
        )

        vlm_region = self._make_doc_region(
            "vlm_a1",
            "D",
            page=1,
            bbox=BBox(x=100, y=200, width=40, height=20),
            role="ANSWER_REGION",
            metadata={
                "question_number": "1",
                "answer_to_question_number": "1",
                "answer_to": "Q1",
                "grounding_status": "GROUNDED",
                "grounded_ocr_region_ids": ["ocr_1"],
                "answer_to_confidence": 0.95,
            },
        )
        vlm_region.confidence = 0.92

        structured = _augment_with_vlm_evidence(structured, SimpleNamespace(regions=[vlm_region]))
        result = structured.answer_regions[0]

        assert result.question_anchor == "1"
        assert result.answer_to == "Q1"
        assert result.answer_to_question_number == "1"
        assert result.answer_to_confidence == 0.95
        assert result.vlm_text == "D"
        assert result.grounding_status == "GROUNDED"

    def test_02_overlap_does_not_discard_vlm_semantics(self):
        ocr_region = AnswerRegion(
            answer_id="ocr_a1",
            question_anchor="Q1",
            pages=[1],
            regions=[Region(page=1, bbox=BBox(x=100, y=200, width=40, height=20))],
            text="B",
            ocr_text="B",
            selected_text="B",
            text_source="OCR",
            grounding_status="GROUNDED",
            grounded_ocr_region_ids=["ocr_a1"],
        )
        structured = StructuredAnswerSheet(
            num_pages=1,
            page_analyses={},
            answer_regions=[ocr_region],
            unanchored_regions=[],
        )

        vlm_region = self._make_doc_region(
            "vlm_1",
            "D",
            page=1,
            bbox=BBox(x=100, y=200, width=40, height=20),
            role="ANSWER_REGION",
            confidence=0.96,
            metadata={
                "question_number": "1",
                "answer_to": "Q1",
                "grounding_status": "GROUNDED",
                "grounded_ocr_region_ids": ["ocr_a1"],
            },
        )

        structured = _augment_with_vlm_evidence(structured, SimpleNamespace(regions=[vlm_region]))
        merged = structured.answer_regions[0]

        assert merged.vlm_text == "D"
        assert merged.ocr_text == "B"
        assert merged.selected_text == "D"
        assert merged.text_source == "VLM_REVIEW_REQUIRED"
        assert merged.answer_to == "Q1"

    def test_03_ocr_and_vlm_disagreement_keeps_both_sources(self):
        structured = StructuredAnswerSheet(
            num_pages=1,
            page_analyses={},
            answer_regions=[
                AnswerRegion(
                    answer_id="ocr_a",
                    question_anchor="Q2",
                    pages=[1],
                    regions=[Region(page=1, bbox=BBox(x=50, y=300, width=50, height=20))],
                    text="B",
                    ocr_text="B",
                    selected_text="B",
                    text_source="OCR",
                    grounding_status="GROUNDED",
                    grounded_ocr_region_ids=["ocr_a"],
                )
            ],
            unanchored_regions=[],
        )
        vlm_region = self._make_doc_region(
            "vlm_a", "D", page=1,
            bbox=BBox(x=50, y=300, width=50, height=20),
            role="ANSWER_REGION",
            confidence=0.90,
            metadata={"question_number": "2", "grounding_status": "GROUNDED", "grounded_ocr_region_ids": ["ocr_a"]},
        )

        structured = _augment_with_vlm_evidence(structured, SimpleNamespace(regions=[vlm_region]))
        region = structured.answer_regions[0]

        assert region.ocr_text == "B"
        assert region.vlm_text == "D"
        assert region.selected_text == "D"
        assert region.text_source == "VLM_REVIEW_REQUIRED"
        assert region.provenance["source"] == "VLM"
        assert region.review_required is True

    def test_04_ungrounded_vlm_result_is_not_verified(self):
        svc = DocumentUnderstandingService.__new__(DocumentUnderstandingService)
        r1 = self._make_doc_region("r_1", "Q1", page=1, bbox=BBox(x=20, y=50, width=80, height=20), role="QUESTION")
        struct = VLMStructureItem(
            role="ANSWER_REGION",
            bbox=BBox(x=100, y=200, width=50, height=25),
            confidence=0.55,
            vlm_text="(D)",
            region_ids=[],
        )
        understanding = VLMPageUnderstanding(page_number=1, structures=[struct], semantic_completeness="AMBIGUOUS")
        all_regions = [r1]

        svc._apply_vlm_page_understandings([understanding], all_regions, [], {})

        synthetic = [r for r in all_regions if r.region_id.startswith("vlm_visual_")][0]
        assert synthetic.verification_state == "UNVERIFIED"
        assert synthetic.metadata["grounding_status"] == "UNGROUNDED"
        assert synthetic.metadata["review_required"] is True or synthetic.metadata["text_source"] == "STRUCTURAL_LABEL_ONLY"

    def test_05_question_number_and_answer_in_separate_table_cells(self):
        structured = StructuredAnswerSheet(
            num_pages=1,
            page_analyses={},
            answer_regions=[],
            unanchored_regions=[],
        )
        vlm_region = self._make_doc_region(
            "table_cell",
            "D",
            page=1,
            bbox=BBox(x=200, y=400, width=35, height=20),
            role="ANSWER_REGION",
            metadata={
                "question_number": "3",
                "answer_to_question_number": "3",
                "answer_to": "Q3",
                "grounding_status": "GROUNDED",
                "grounded_ocr_region_ids": ["cell_3"],
                "answer_to_confidence": 0.9,
            },
        )

        structured = _augment_with_vlm_evidence(structured, SimpleNamespace(regions=[vlm_region]))
        region = structured.answer_regions[0]

        assert region.question_anchor == "3"
        assert region.answer_to == "Q3"
        assert region.answer_to_question_number == "3"

    def test_06_multiple_answer_regions_are_kept_separate(self):
        structured = StructuredAnswerSheet(
            num_pages=1,
            page_analyses={},
            answer_regions=[],
            unanchored_regions=[],
        )
        result = SimpleNamespace(regions=[
            self._make_doc_region("v1", "D", page=1, bbox=BBox(x=10, y=50, width=30, height=15), role="ANSWER_REGION", metadata={"question_number": "1"}),
            self._make_doc_region("v2", "B", page=1, bbox=BBox(x=10, y=120, width=30, height=15), role="ANSWER_REGION", metadata={"question_number": "2"}),
        ])

        structured = _augment_with_vlm_evidence(structured, result)
        assert len(structured.answer_regions) == 2
        assert {r.question_anchor for r in structured.answer_regions} == {"1", "2"}
        assert len({r.answer_id for r in structured.answer_regions}) == 2
