"""
Regression tests for Fix #1:
Preventing false QUESTION creation from isolated OCR fragments containing interrogative keywords.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import types
from unittest.mock import MagicMock

# Mock sklearn and sentence_transformers so DocumentUnderstandingService can be imported in test environment
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

from app.models.schemas import DocumentRegion, BBox
from app.services.document_understanding_service import DocumentUnderstandingService



class TestDeterministicHypothesisClassification:

    def _make_service(self):
        service = DocumentUnderstandingService.__new__(DocumentUnderstandingService)
        return service

    def _classify_text(self, text: str, page_width: float = 600.0, page_height: float = 800.0) -> DocumentRegion:
        service = self._make_service()
        # Position in mid-page so header/footer layout analyzer does not trigger
        reg = DocumentRegion(
            region_id="reg_test_1",
            page=1,
            text=text,
            bbox=BBox(x=50.0, y=200.0, width=300.0, height=20.0),
            region_type="UNKNOWN",
            confidence=0.5,
            evidence=[],
            relationships=[],
            uncertainty=0.0,
            classification_conflict=False,
            conflicting_hypotheses=[],
            verification_state="UNVERIFIED",
            metadata={"modality": "text", "role": "paragraph"},
        )
        service._analyze_region_hypotheses(reg, page_width, page_height)
        return reg

    # CASE A: Isolated fragment "What"
    def test_case_a_isolated_what_is_not_question(self):
        reg = self._classify_text("What")
        assert reg.region_type != "QUESTION", f"Expected NOT QUESTION, got {reg.region_type}"
        assert reg.region_type == "UNKNOWN"

    # CASE B: Isolated fragment "which"
    def test_case_b_isolated_which_is_not_question(self):
        reg = self._classify_text("which")
        assert reg.region_type != "QUESTION", f"Expected NOT QUESTION, got {reg.region_type}"
        assert reg.region_type == "UNKNOWN"

    # CASE C: Isolated fragment "List"
    def test_case_c_isolated_list_is_not_question(self):
        reg = self._classify_text("List")
        assert reg.region_type != "QUESTION", f"Expected NOT QUESTION, got {reg.region_type}"
        assert reg.region_type == "UNKNOWN"

    # CASE D: Isolated fragment "Why"
    def test_case_d_isolated_why_is_not_question(self):
        reg = self._classify_text("Why")
        assert reg.region_type != "QUESTION", f"Expected NOT QUESTION, got {reg.region_type}"
        assert reg.region_type == "UNKNOWN"

    # Additional fragments
    def test_isolated_explain_is_not_question(self):
        reg = self._classify_text("Explain")
        assert reg.region_type != "QUESTION"
        assert reg.region_type == "UNKNOWN"

    def test_isolated_describe_is_not_question(self):
        reg = self._classify_text("Describe")
        assert reg.region_type != "QUESTION"
        assert reg.region_type == "UNKNOWN"

    # CASE E: Unnumbered question without structural numbering
    # Deterministic parser leaves it UNKNOWN (VLM is the visual semantic authority)
    def test_case_e_unnumbered_question_remains_unknown_deterministically(self):
        reg = self._classify_text("What is photosynthesis?")
        assert reg.region_type != "QUESTION", "Unnumbered text must not be deterministically promoted to QUESTION"
        assert reg.region_type == "UNKNOWN"

    # CASE F: Clearly numbered question with interrogative word
    def test_case_f_numbered_question_with_interrogative_is_question(self):
        reg = self._classify_text("12. What is photosynthesis?")
        assert reg.region_type == "QUESTION"
        assert reg.confidence >= 0.90
        signal_types = [ev.signal_type for ev in reg.evidence]
        assert "numbering_pattern" in signal_types
        assert "question_interrogative" in signal_types

    # Numbered question without interrogative word
    def test_numbered_question_without_interrogative_is_question(self):
        reg = self._classify_text("1. Write briefly about the nitrogen cycle.")
        assert reg.region_type == "QUESTION"
        assert reg.confidence >= 0.80

    def test_q_prefix_numbered_question_is_question(self):
        reg = self._classify_text("Q2. Define activation function.")
        assert reg.region_type == "QUESTION"
        assert reg.confidence >= 0.80

    # CASE G: Option with standard text
    def test_case_g_standard_option_is_option(self):
        reg = self._classify_text("(A) Ethane")
        assert reg.region_type == "OPTION"
        assert reg.confidence >= 0.85

    # Option containing interrogative keyword
    def test_option_with_interrogative_word_remains_option(self):
        reg = self._classify_text("(A) Which of the following is correct?")
        assert reg.region_type == "OPTION", f"Expected OPTION, got {reg.region_type}"

    # Subquestion
    def test_subquestion_marker_is_subquestion(self):
        reg = self._classify_text("(i) State Ohm's law.")
        assert reg.region_type == "SUBQUESTION"

    def test_subquestion_alphabet_marker_is_subquestion(self):
        reg = self._classify_text("(a) Calculate the molecular weight.")
        assert reg.region_type == "SUBQUESTION"

    # Instructions containing interrogative keyword
    def test_instruction_with_interrogative_word_is_instruction(self):
        reg = self._classify_text("Instructions: Explain any four questions.")
        assert reg.region_type == "INSTRUCTION"

    # Section header
    def test_section_header_is_section_header(self):
        reg = self._classify_text("SECTION - B")
        assert reg.region_type == "SECTION_HEADER"
