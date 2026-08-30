#!/usr/bin/env python
"""
FIX #2 FOCUSED GROUNDING TESTS

Tests for geometry-based grounding algorithm:
1. One VLM bbox → one OCR region
2. One VLM bbox → multiple OCR regions
3. OCR fragments split across multiple lines
4. Partial overlap
5. Containment
6. No matching OCR region
7. Different coordinate scales
8. MCQ option grounding
9. Q1 + 1(a)-1(j) grounding
10. Administrative document handling
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from app.models.schemas import Block, BBox, DocumentRegion, VLMStructureItem
from app.services.document_understanding_service import DocumentUnderstandingService


class TestGroundingAlgorithm(unittest.TestCase):
    """Test the _ground_structure_to_ocr algorithm."""

    def setUp(self):
        self.service = DocumentUnderstandingService()

    def test_01_single_bbox_single_region(self):
        """Test 1: One VLM bbox perfectly matches one OCR region."""
        # OCR region
        region = DocumentRegion(
            region_id="b1",
            page=1,
            text="Question 1",
            bbox=BBox(x=100, y=100, width=200, height=30),
        )
        
        # VLM visual bbox matching the OCR region
        structure = VLMStructureItem(
            role="QUESTION",
            bbox=BBox(x=100, y=100, width=200, height=30),
            confidence=0.95,
            reasoning="Perfect match",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=[region],
            page_number=1,
        )
        
        self.assertEqual(len(grounded_ids), 1)
        self.assertEqual(grounded_ids[0], "b1")
        self.assertEqual(status, "GROUNDED")
        self.assertEqual(text, "Question 1")
        print("[OK] Test 1: Single bbox -> single region: PASS")

    def test_02_single_bbox_multiple_regions(self):
        """Test 2: One VLM bbox spans multiple OCR regions."""
        # OCR regions representing a multi-line question
        regions = [
            DocumentRegion(
                region_id="b1",
                page=1,
                text="Question 1:",
                bbox=BBox(x=100, y=100, width=150, height=20),
            ),
            DocumentRegion(
                region_id="b2",
                page=1,
                text="(a) What is X?",
                bbox=BBox(x=110, y=125, width=180, height=20),
            ),
            DocumentRegion(
                region_id="b3",
                page=1,
                text="(b) Define Y?",
                bbox=BBox(x=110, y=150, width=170, height=20),
            ),
        ]
        
        # VLM visual bbox covering all three lines
        structure = VLMStructureItem(
            role="QUESTION",
            bbox=BBox(x=100, y=100, width=200, height=80),
            confidence=0.90,
            reasoning="Multi-line question",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=regions,
            page_number=1,
        )
        
        self.assertEqual(len(grounded_ids), 3)
        self.assertIn("b1", grounded_ids)
        self.assertIn("b2", grounded_ids)
        self.assertIn("b3", grounded_ids)
        self.assertEqual(status, "GROUNDED")
        self.assertIn("Question 1:", text)
        self.assertIn("(a)", text)
        self.assertIn("(b)", text)
        print("✓ Test 2: Single bbox → multiple regions: PASS")

    def test_03_weak_overlap(self):
        """Test 3: VLM bbox with tiny accidental contact - below threshold."""
        region = DocumentRegion(
            region_id="b1",
            page=1,
            text="Section A",
            bbox=BBox(x=100, y=100, width=200, height=30),
        )
        
        # VLM bbox barely touching at corner (1 pixel contact, ~0.3% overlap)
        structure = VLMStructureItem(
            role="SECTION_HEADER",
            bbox=BBox(x=299, y=129, width=100, height=30),
            confidence=0.50,
            reasoning="Tiny corner touch",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=[region],
            page_number=1,
        )
        
        # Should NOT ground with only corner contact (well below 0.15 threshold)
        self.assertEqual(len(grounded_ids), 0)
        self.assertEqual(status, "UNGROUNDED")
        print("✓ Test 3: Tiny corner touch filtered out: PASS")

    def test_04_containment(self):
        """Test 4: OCR region fully contained in VLM bbox."""
        region = DocumentRegion(
            region_id="b1",
            page=1,
            text="Option A",
            bbox=BBox(x=120, y=110, width=100, height=20),
        )
        
        # VLM bbox fully contains the OCR region
        structure = VLMStructureItem(
            role="OPTION",
            bbox=BBox(x=100, y=100, width=200, height=50),
            confidence=0.92,
            reasoning="Containment",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=[region],
            page_number=1,
        )
        
        self.assertEqual(len(grounded_ids), 1)
        self.assertEqual(grounded_ids[0], "b1")
        self.assertEqual(status, "GROUNDED")
        print("✓ Test 4: Containment: PASS")

    def test_05_no_matching_region(self):
        """Test 5: VLM bbox has no matching OCR region."""
        regions = [
            DocumentRegion(
                region_id="b1",
                page=1,
                text="Question 1",
                bbox=BBox(x=100, y=100, width=200, height=30),
            ),
        ]
        
        # VLM bbox completely separate from OCR region
        structure = VLMStructureItem(
            role="QUESTION",
            bbox=BBox(x=500, y=500, width=200, height=30),
            confidence=0.80,
            reasoning="No overlap",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=regions,
            page_number=1,
        )
        
        self.assertEqual(len(grounded_ids), 0)
        self.assertEqual(status, "UNGROUNDED")
        self.assertEqual(text, "")
        print("✓ Test 5: No matching region: PASS")

    def test_06_multiple_candidates_filtering(self):
        """Test 6: Multiple overlapping regions, best ones selected."""
        regions = [
            DocumentRegion(
                region_id="b1",
                page=1,
                text="Main text",
                bbox=BBox(x=100, y=100, width=400, height=30),  # High overlap
            ),
            DocumentRegion(
                region_id="b2",
                page=1,
                text="Small text",
                bbox=BBox(x=500, y=100, width=50, height=30),  # Minimal overlap
            ),
            DocumentRegion(
                region_id="b3",
                page=1,
                text="Medium text",
                bbox=BBox(x=200, y=140, width=200, height=30),  # Some overlap
            ),
        ]
        
        structure = VLMStructureItem(
            role="QUESTION",
            bbox=BBox(x=100, y=100, width=400, height=70),
            confidence=0.90,
            reasoning="Multiple candidates",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=regions,
            page_number=1,
        )
        
        # Should include high-overlap and medium-overlap, exclude low-overlap
        self.assertIn("b1", grounded_ids)
        self.assertIn("b3", grounded_ids)
        self.assertNotIn("b2", grounded_ids)  # Too little overlap
        self.assertEqual(status, "GROUNDED")
        print("✓ Test 6: Multiple candidates filtering: PASS")

    def test_07_axis_alignment(self):
        """Test 7: VLM bbox aligned on one axis but not the other."""
        # OCR region at (100, 100)
        region = DocumentRegion(
            region_id="b1",
            page=1,
            text="Text",
            bbox=BBox(x=100, y=100, width=100, height=30),
        )
        
        # VLM bbox aligned horizontally but offset vertically
        structure = VLMStructureItem(
            role="QUESTION",
            bbox=BBox(x=100, y=150, width=100, height=30),
            confidence=0.75,
            reasoning="Axis misalignment",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=[region],
            page_number=1,
        )
        
        # Should not match (0 overlap)
        self.assertEqual(len(grounded_ids), 0)
        self.assertEqual(status, "UNGROUNDED")
        print("✓ Test 7: Axis alignment: PASS")

    def test_08_mcq_option_grounding(self):
        """Test 8: MCQ option grounding within section."""
        regions = [
            DocumentRegion(
                region_id="s1",
                page=1,
                text="Section A: Multiple Choice",
                bbox=BBox(x=50, y=50, width=300, height=25),
            ),
            DocumentRegion(
                region_id="q1",
                page=1,
                text="1. Which is correct?",
                bbox=BBox(x=60, y=85, width=280, height=25),
            ),
            DocumentRegion(
                region_id="opt_a",
                page=1,
                text="(A) Option A",
                bbox=BBox(x=70, y=115, width=100, height=20),
            ),
            DocumentRegion(
                region_id="opt_b",
                page=1,
                text="(B) Option B",
                bbox=BBox(x=70, y=140, width=100, height=20),
            ),
            DocumentRegion(
                region_id="opt_c",
                page=1,
                text="(C) Option C",
                bbox=BBox(x=70, y=165, width=100, height=20),
            ),
        ]
        
        # VLM visual bbox for the options
        structure = VLMStructureItem(
            role="OPTION",
            bbox=BBox(x=65, y=110, width=110, height=80),
            confidence=0.88,
            reasoning="MCQ options",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=regions,
            page_number=1,
        )
        
        # Should ground to all three options
        self.assertIn("opt_a", grounded_ids)
        self.assertIn("opt_b", grounded_ids)
        self.assertIn("opt_c", grounded_ids)
        self.assertEqual(status, "GROUNDED")
        print("✓ Test 8: MCQ option grounding: PASS")

    def test_09_subquestion_hierarchy(self):
        """Test 9: Q1 with subquestions 1(a)-1(c) grounding."""
        regions = [
            DocumentRegion(
                region_id="q1",
                page=1,
                text="1. Explain the concept:",
                bbox=BBox(x=50, y=100, width=300, height=25),
            ),
            DocumentRegion(
                region_id="sq_a",
                page=1,
                text="(a) Part A explanation",
                bbox=BBox(x=60, y=135, width=280, height=25),
            ),
            DocumentRegion(
                region_id="sq_b",
                page=1,
                text="(b) Part B explanation",
                bbox=BBox(x=60, y=170, width=280, height=25),
            ),
            DocumentRegion(
                region_id="sq_c",
                page=1,
                text="(c) Part C explanation",
                bbox=BBox(x=60, y=205, width=280, height=25),
            ),
        ]
        
        # VLM visual bbox for entire question with subparts
        structure = VLMStructureItem(
            role="QUESTION",
            bbox=BBox(x=45, y=95, width=310, height=140),
            confidence=0.92,
            reasoning="Q1 with subquestions",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=structure,
            page_regions=regions,
            page_number=1,
        )
        
        # Should ground to all parts
        self.assertEqual(len(grounded_ids), 4)
        self.assertIn("q1", grounded_ids)
        self.assertIn("sq_a", grounded_ids)
        self.assertIn("sq_b", grounded_ids)
        self.assertIn("sq_c", grounded_ids)
        self.assertEqual(status, "GROUNDED")
        print("✓ Test 9: Subquestion hierarchy: PASS")

    def test_10_administrative_vs_question(self):
        """Test 10: Administrative metadata not grounded as questions."""
        regions = [
            DocumentRegion(
                region_id="adm1",
                page=1,
                text="Course: Computer Science 101",
                bbox=BBox(x=50, y=20, width=300, height=20),
            ),
            DocumentRegion(
                region_id="adm2",
                page=1,
                text="Time: 3 hours",
                bbox=BBox(x=50, y=45, width=200, height=20),
            ),
            DocumentRegion(
                region_id="real_q",
                page=1,
                text="1. Define algorithm",
                bbox=BBox(x=50, y=150, width=350, height=25),
            ),
        ]
        
        # VLM correctly identifies administrative content
        admin_structure = VLMStructureItem(
            role="INSTRUCTION",
            bbox=BBox(x=45, y=15, width=310, height=55),
            confidence=0.95,
            reasoning="Administrative header",
        )
        
        grounded_ids, status, text = self.service._ground_structure_to_ocr(
            structure=admin_structure,
            page_regions=regions,
            page_number=1,
        )
        
        # Should ground to administrative regions only
        self.assertEqual(len(grounded_ids), 2)
        self.assertIn("adm1", grounded_ids)
        self.assertIn("adm2", grounded_ids)
        self.assertNotIn("real_q", grounded_ids)
        print("✓ Test 10: Administrative vs question distinction: PASS")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGroundingAlgorithm)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n" + "="*60)
        print("✓ ALL FOCUSED GROUNDING TESTS PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
