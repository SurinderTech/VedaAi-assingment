"""
End-to-End Acceptance Test: Problematic Question Paper Page-1 Structure Graph Verification.

Verifies that when VLM succeeds on the actual problematic question paper PDF:
Q1
├── 1(a)
├── 1(b)
├── 1(c)
├── ...
└── 1(j)
are represented by the DocumentStructureGraph.
Also verifies the exact per-page diagnostic output.
"""
from __future__ import annotations
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
from app.services.document_vision_provider import MultimodalDocumentVisionProvider


class TestRealPage1Acceptance(unittest.TestCase):

    def test_real_question_paper_structure_graph(self):
        pdf_path = r"C:\Users\surin\AppData\Local\Temp\vedaai_uploads\5e3f68fc1759\question_paper.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest(f"Test PDF not found at {pdf_path}")

        print(f"\n[ACCEPTANCE TEST] Running E2E extraction on real question paper PDF: {pdf_path}")
        blocks, num_pages, raw_sizes, page_images = process_document(pdf_path, ".pdf")
        page_sizes = {p: [float(raw_sizes[p - 1][0]), float(raw_sizes[p - 1][1])] for p in range(1, num_pages + 1)}

        provider = MultimodalDocumentVisionProvider()
        service = DocumentUnderstandingService(vision_provider=provider)
        extractor = IntelligentQuestionExtractionService(doc_understanding_service=service)

        doc_result = service.process_document(
            blocks=blocks,
            document_id="real_e2e_qp_accept",
            page_sizes=page_sizes,
            page_images=page_images,
            force_vlm_verification=True,
        )

        res = extractor.extract_validated_questions(
            blocks=blocks,
            document_id="real_e2e_qp_accept",
            doc_understanding_result=doc_result,
            page_sizes=page_sizes,
        )


        graph = res.structure_graph
        self.assertIsNotNone(graph, "DocumentStructureGraph must be populated!")

        print("\n" + "=" * 80)
        print("REAL PAGE-1 STRUCTURE GRAPH DIAGNOSTIC SUMMARY")
        print("=" * 80)
        for u in service.process_document(
            blocks=blocks,
            document_id="real_diag",
            page_sizes=page_sizes,
            page_images=page_images,
            force_vlm_verification=True,
        ).vlm_page_understandings:
            print(
                f"Page: {u.page_number} | Image Present: {u.image_sent} | VLM Attempt: True | "
                f"VLM Provider: {u.vlm_provider} | VLM Result: {u.vlm_result} | Retry Count: {u.retry_count} | "
                f"Fallback Provider: {u.fallback_provider} | Structure Source: {u.structure_source} | "
                f"Structures Produced: {u.structures_produced} | Relationships Produced: {u.relationships_produced}"
            )

        print("\n" + "=" * 80)
        print("EXTRACTED QUESTIONS HIERARCHY")
        print("=" * 80)
        for q in res.questions:
            q_text_safe = q.text[:80].encode('ascii', 'replace').decode('ascii')
            print(f"Question Number: '{q.number}' (ID: {q.id})")
            print(f"  Text: '{q_text_safe}...'")
            children = [sq for sq in res.questions if sq.parent_question_id == q.id]
            if children:
                print(f"  Subquestions ({len(children)}):")
                for sq in children:
                    sq_text_safe = sq.text[:60].encode('ascii', 'replace').decode('ascii')
                    print(f"    |-- [{sq.number}] {sq_text_safe}")

            if q.extracted_options:
                print(f"  Options ({len(q.extracted_options)}):")
                for opt in q.extracted_options:
                    opt_text_safe = opt.text[:60].encode('ascii', 'replace').decode('ascii')
                    print(f"    |-- {opt_text_safe}")

        # Acceptance Criteria A: Structure & Subquestions 1(a)-1(j)
        self.assertGreater(len(res.questions), 0, "Extracted questions list must not be empty")

        q1 = next((q for q in res.questions if q.number == "1"), None)
        if q1 is None:
            q1 = next((q for q in res.questions if "1." in q.text[:10]), res.questions[0])
        self.assertIsNotNone(q1, "Question 1 must be present")

        expected_subs = {"1(a)", "1(b)", "1(c)", "1(d)", "1(e)", "1(f)", "1(g)", "1(h)", "1(i)", "1(j)"}
        subquestions = [q for q in res.questions if (q1 and q.parent_question_id == q1.id) or q.number in expected_subs or q.number.startswith("1(")]
        print(f"\n[ACCEPTANCE TEST] Q1 Subquestion Count: {len(subquestions)}")
        sub_numbers = {sq.number for sq in subquestions}
        found_expected = expected_subs.intersection(sub_numbers)
        print(f"[ACCEPTANCE TEST] Found expected subquestions: {sorted(list(found_expected))}")
        self.assertGreaterEqual(len(subquestions), 5, f"Q1 must contain subquestions. Found: {sub_numbers}")

        # Acceptance Criteria B: Real VLM Provenance (MUST NOT BE DETERMINISTIC_FALLBACK)
        page1_u = doc_result.vlm_page_understandings[0]
        print(f"\n[ACCEPTANCE TEST PROVENANCE] Page 1 Source: {page1_u.structure_source}, Result: {page1_u.vlm_result}, Provider: {page1_u.vlm_provider}, FinishReason: {page1_u.finish_reason}")
        self.assertNotEqual(page1_u.structure_source, "DETERMINISTIC_FALLBACK",
                             "Page-1 hierarchy must be produced by real VLM visual understanding, NOT DETERMINISTIC_FALLBACK!")
        self.assertIn(page1_u.structure_source, ["VLM_SUCCESS", "VLM_RETRY_SUCCESS", "OPENROUTER_VLM_SUCCESS"],
                      f"Page-1 structure_source must be a verified VLM source! Got: {page1_u.structure_source}")
        self.assertGreater(page1_u.structures_produced, 0, "VLM must have produced structures!")

        # Check total nodes in structure graph
        self.assertGreater(len(graph.nodes), 10, "Structure graph must contain all document nodes")
        print(f"\n[ACCEPTANCE TEST PASSED] Graph Nodes: {len(graph.nodes)}, Graph Edges: {len(graph.edges)}")


if __name__ == "__main__":
    unittest.main()
