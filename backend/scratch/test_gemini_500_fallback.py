"""
Regression Test: Gemini Page-1 HTTP 500 Failure, Retry Backoff & Explicit Provenance.

Verifies:
1. Gemini transient HTTP 500 status code triggers retry with backoff.
2. Structure source is explicitly reported as VLM_RETRY_SUCCESS when retry succeeds.
3. Structure source is explicitly reported as DETERMINISTIC_FALLBACK when VLM fails completely.
4. No structures are falsely attributed to VLM when Gemini fails.
"""
from __future__ import annotations
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import Block, BBox, DocumentUnderstandingResult
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.services.llm_provider import LLMError, VLMError


class TestGemini500Fallback(unittest.TestCase):

    def setUp(self):
        self.doc_service = DocumentUnderstandingService()

    @patch("app.services.llm_provider.httpx.AsyncClient.post")
    def test_gemini_500_transient_retry_success(self, mock_post):
        """Tests that Gemini 500 transient error triggers retry and reports VLM_RETRY_SUCCESS."""
        mock_r500 = MagicMock()
        mock_r500.status_code = 500
        mock_r500.text = "Internal Server Error"

        mock_r200 = MagicMock()
        mock_r200.status_code = 200
        mock_r200.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"page_purpose": "QUESTION_PAGE", "document_purpose": "EXAMINATION_PAPER", "structures": [{"region_ids": ["b1"], "role": "QUESTION", "display_number": "1", "reasoning": "Q1 text", "confidence": 0.95}], "relationships": []}'
                    }]
                }
            }]
        }
        mock_post.side_effect = [mock_r500, mock_r200]

        dummy_img = Image.new("RGB", (1000, 1400), color="white")
        blocks = [Block(id="b1", text="1. What is machine learning?", confidence=0.95, bbox=BBox(x=100, y=100, width=400, height=30), page=1)]

        result = self.doc_service.process_document(
            blocks=blocks,
            document_id="doc_retry_test",
            page_sizes={1: [1000.0, 1400.0]},
            page_images={1: dummy_img},
            force_vlm_verification=True,
        )

        self.assertEqual(len(result.vlm_page_understandings), 1)
        u = result.vlm_page_understandings[0]
        print(f"\n[REGRESSION TEST] Retries: {u.retry_count}, Provenance: {u.structure_source}")
        self.assertEqual(u.vlm_result, "SUCCESS")
        self.assertEqual(u.structure_source, "VLM_RETRY_SUCCESS")
        self.assertGreater(u.retry_count, 0)
        self.assertEqual(len(u.structures), 1)

    @patch("app.services.llm_provider.httpx.AsyncClient.post")
    def test_gemini_500_complete_failure_deterministic_fallback(self, mock_post):
        """Tests that when Gemini fails across all attempts, structure source is DETERMINISTIC_FALLBACK."""
        mock_r500 = MagicMock()
        mock_r500.status_code = 500
        mock_r500.text = "Internal Server Error"
        mock_post.return_value = mock_r500

        dummy_img = Image.new("RGB", (1000, 1400), color="white")
        blocks = [Block(id="b1", text="1. What is deep learning?", confidence=0.95, bbox=BBox(x=100, y=100, width=400, height=30), page=1)]

        result = self.doc_service.process_document(
            blocks=blocks,
            document_id="doc_fail_test",
            page_sizes={1: [1000.0, 1400.0]},
            page_images={1: dummy_img},
            force_vlm_verification=True,
        )

        self.assertEqual(len(result.vlm_page_understandings), 1)
        u = result.vlm_page_understandings[0]
        print(f"\n[REGRESSION TEST] Provenance on Failure: {u.structure_source}, VLM Result: {u.vlm_result}")
        self.assertEqual(u.vlm_result, "FAILED")
        self.assertEqual(u.structure_source, "DETERMINISTIC_FALLBACK")
        self.assertEqual(len(u.structures), 0)


if __name__ == "__main__":
    unittest.main()
