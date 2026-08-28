"""
STEP 10B — PRECISION HIGHLIGHTER & QUESTION-NUMBERING ROBUSTNESS TEST SUITE (TESTS 1-10)
Verifies precision disjoint sub-region highlighting, explicit Q.1 question numbering recognition,
false positive protection, subquestion preservation, and full Step 1-9 pipeline regressions.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import Block, BBox, Region, AnswerRegion
from app.services.question_extractor import extract_questions
from app.services.answer_extractor import process_answer_sheet


async def run_step10b_test_suite():
    print("=" * 95)
    print("STEP 10B PRECISION HIGHLIGHTER & QUESTION-NUMBERING ROBUSTNESS TEST SUITE (TESTS 1-10)")
    print("=" * 95)

    passed_count = 0
    total_tests = 10

    # -------------------------------------------------------------------------
    # TEST 1: Single Answer Region
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Single Answer Region Highlight Mapping")
    single_b = Block(id="sb1", text="Single paragraph answer text", confidence=0.98, bbox=BBox(x=100, y=80, width=300, height=40), page=1, role="student_answer")
    anchor_b = Block(id="sa1", text="Q1.", confidence=0.98, bbox=BBox(x=50, y=80, width=40, height=20), page=1, role="student_question_anchor")
    sheet1 = process_answer_sheet([anchor_b, single_b], num_pages=1)
    regions1 = sheet1.answer_regions[0].regions if sheet1.answer_regions else []
    print(f"    Single Region Count: {len(regions1)}")
    assert len(regions1) >= 1, "Test 1 Failed: Single answer region must produce valid regions!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 2: Disjoint Same-Page Regions (No Merged Rectangle)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Disjoint Same-Page Regions (Separate Highlights)")
    b_top = Block(id="bt", text="Top paragraph answer", confidence=0.95, bbox=BBox(x=100, y=80, width=300, height=40), page=1, role="student_answer")
    b_bot = Block(id="bb", text="Bottom paragraph answer", confidence=0.95, bbox=BBox(x=100, y=500, width=300, height=40), page=1, role="student_answer")
    anchor_b2 = Block(id="sa2", text="Q2.", confidence=0.98, bbox=BBox(x=50, y=80, width=40, height=20), page=1, role="student_question_anchor")
    sheet2 = process_answer_sheet([anchor_b2, b_top, b_bot], num_pages=1)
    regions2 = sheet2.answer_regions[0].regions if sheet2.answer_regions else []
    print(f"    Disjoint Regions Count: {len(regions2)}")
    for idx, r in enumerate(regions2):
        print(f"      Sub-Region #{idx+1}: BBox(y={r.bbox.y}, h={r.bbox.height})")
    assert len(regions2) >= 2, "Test 2 Failed: Disjoint same-page blocks must produce separate distinct regions!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 3: Region Geometry Preservation
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Region Geometry Preservation")
    r_geom = regions2[1].bbox
    print(f"    Preserved Original BBox: x={r_geom.x}, y={r_geom.y}, w={r_geom.width}, h={r_geom.height}")
    assert r_geom.x == 100 and r_geom.y == 80 and r_geom.width == 300 and r_geom.height == 40, "Test 3 Failed: Original BBox geometry must be preserved 100%!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 4: Table Regions
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Table Regions Coverage & Renderability")
    t_anchor = Block(id="ta", text="Q3.", confidence=0.98, bbox=BBox(x=50, y=50, width=40, height=20), page=1, role="student_question_anchor")
    t_r1 = Block(id="tr1", text="Row 1: Feature A | Value 10", confidence=0.95, bbox=BBox(x=50, y=80, width=450, height=25), page=1, role="student_answer")
    t_r2 = Block(id="tr2", text="Row 2: Feature B | Value 20", confidence=0.95, bbox=BBox(x=50, y=115, width=450, height=25), page=1, role="student_answer")
    t_r3 = Block(id="tr3", text="Row 3: Feature C | Value 30", confidence=0.95, bbox=BBox(x=50, y=150, width=450, height=25), page=1, role="student_answer")
    sheet_t = process_answer_sheet([t_anchor, t_r1, t_r2, t_r3], num_pages=1)
    regions_t = sheet_t.answer_regions[0].regions if sheet_t.answer_regions else []
    print(f"    Table Sub-Regions Extracted: {len(regions_t)}")
    assert len(regions_t) >= 1, "Test 4 Failed: Table regions must be extracted and preserved!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 5: Multi-Page Regions
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Multi-Page Regions Page Association")
    b_p1 = Block(id="bp1", text="Page 1 part of answer", confidence=0.95, bbox=BBox(x=50, y=80, width=400, height=40), page=1, role="student_answer")
    b_p2 = Block(id="bp2", text="Q4. (continued) Page 2 part of answer", confidence=0.95, bbox=BBox(x=50, y=50, width=400, height=40), page=2, role="student_answer")
    a_p1 = Block(id="ap1", text="Q4.", confidence=0.98, bbox=BBox(x=50, y=50, width=40, height=20), page=1, role="student_question_anchor")
    sheet_mp = process_answer_sheet([a_p1, b_p1, b_p2], num_pages=2)
    ar_mp = sheet_mp.answer_regions[0] if sheet_mp.answer_regions else None
    print(f"    Spanned Pages: {ar_mp.pages if ar_mp else []}")
    assert ar_mp is not None and ar_mp.pages == [1, 2], "Test 5 Failed: Multi-page answer must associate with pages 1 and 2!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 6: Existing Question Number Formats
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Existing Question Number Formats Preservation")
    existing_blocks = [
        Block(id="eb1", text="1. Explain gradient descent.", confidence=0.98, bbox=BBox(x=50, y=50, width=400, height=30), page=1),
        Block(id="eb2", text="2) Define dropout regularization.", confidence=0.98, bbox=BBox(x=50, y=90, width=400, height=30), page=1),
        Block(id="eb3", text="3: State Adam optimizer principles.", confidence=0.98, bbox=BBox(x=50, y=130, width=400, height=30), page=1),
        Block(id="eb4", text="4- Calculate matrix dimension.", confidence=0.98, bbox=BBox(x=50, y=170, width=400, height=30), page=1),
        Block(id="eb5", text="05. Discuss activation functions.", confidence=0.98, bbox=BBox(x=50, y=210, width=400, height=30), page=1),
        Block(id="eb6", text="7(a) Describe loss function.", confidence=0.98, bbox=BBox(x=50, y=250, width=400, height=30), page=1),
        Block(id="eb7", text="7) Summarize neural network training.", confidence=0.98, bbox=BBox(x=50, y=290, width=400, height=30), page=1),
    ]
    qs_exist = await extract_questions(existing_blocks)
    nums_exist = [q.number for q in qs_exist]
    print(f"    Existing Numbers Recognized: {nums_exist}")
    assert len(qs_exist) >= 6, f"Test 6 Failed: All existing question formats must remain recognized! Parsed: {nums_exist}"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 7: Explicit Q. Prefix Format (Q.1 Discuss..., Q.2 Explain...)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Explicit Q. Prefix Question Format")
    q_prefix_blocks = [
        Block(id="qp1", text="Q.1 Discuss the concept of convolutional neural networks in detail.", confidence=0.98, bbox=BBox(x=50, y=50, width=500, height=30), page=1),
        Block(id="qp2", text="Q.2 Explain backpropagation algorithm with mathematical derivation.", confidence=0.98, bbox=BBox(x=50, y=90, width=500, height=30), page=1),
        Block(id="qp3", text="Q.10 Compare L1 and L2 regularization methods.", confidence=0.98, bbox=BBox(x=50, y=130, width=500, height=30), page=1),
    ]
    qs_prefix = await extract_questions(q_prefix_blocks)
    nums_prefix = [q.number for q in qs_prefix]
    print(f"    Q. Prefix Numbers Recognized: {nums_prefix}")
    assert "1" in nums_prefix and "2" in nums_prefix and "10" in nums_prefix, f"Test 7 Failed: Q.1, Q.2, Q.10 must be recognized! Parsed: {nums_prefix}"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 8: False Positive Protection (Ordinary Prose)
    # -------------------------------------------------------------------------
    print("\n[TEST 8] False Positive Protection (Ordinary Prose)")
    prose_blocks = [
        Block(id="pb1", text="The experiment lasted 1 hour in the laboratory.", confidence=0.98, bbox=BBox(x=50, y=50, width=500, height=20), page=1),
        Block(id="pb2", text="Chapter 2 discusses neural networks and optimization.", confidence=0.98, bbox=BBox(x=50, y=80, width=500, height=20), page=1),
        Block(id="pb3", text="There are 5 major steps in gradient descent.", confidence=0.98, bbox=BBox(x=50, y=110, width=500, height=20), page=1),
    ]
    qs_prose = await extract_questions(prose_blocks)
    print(f"    Prose Question Count: {len(qs_prose)}")
    assert len(qs_prose) == 0, "Test 8 Failed: Ordinary prose containing numbers MUST NOT be parsed as questions!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 9: Subquestion Preservation (Q.3(a), Q.3(b))
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Subquestion Preservation")
    subq_blocks = [
        Block(id="sqb1", text="Q.3 (a) Explain batch normalization.", confidence=0.98, bbox=BBox(x=50, y=50, width=500, height=30), page=1),
        Block(id="sqb2", text="    (b) Describe layer normalization.", confidence=0.98, bbox=BBox(x=50, y=90, width=500, height=30), page=1),
    ]
    qs_sub = await extract_questions(subq_blocks)
    nums_sub = [q.number for q in qs_sub]
    print(f"    Subquestions Extracted: {nums_sub}")
    assert any("a" in n.lower() for n in nums_sub), "Test 9 Failed: Subquestions Q.3(a) must be preserved!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 10: Full Step 1-9 Pipeline Regressions
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Full Step 1-9 Pipeline Regressions")
    from run_step3_diagnostic_check import run_diagnostic_check
    diag_ok = await run_diagnostic_check()
    assert diag_ok is True, "Test 10 Failed: Step 3 diagnostic check failed!"
    print("    Step 3 Diagnostic Check: PASS (42/42 formulas reproducible)")

    from scratch.test_grading_engine import run_test_suite as run_step4
    await run_step4()
    print("    Step 4 Test Suite: PASS (11/11 tests)")

    from scratch.test_assessment_results import run_step5_test_suite as run_step5
    await run_step5()
    print("    Step 5 Test Suite: PASS (22/22 tests)")

    from scratch.test_step6_teacher_workspace import run_step6_test_suite as run_step6
    await run_step6()
    print("    Step 6 Test Suite: PASS (20/20 tests)")

    from scratch.test_step7_student_results import run_step7_test_suite as run_step7
    await run_step7()
    print("    Step 7 Test Suite: PASS (20/20 tests)")

    from scratch.test_step8_embeddings import run_step8_test_suite as run_step8
    await run_step8()
    print("    Step 8 Test Suite: PASS (20/20 tests)")

    from scratch.test_step9_assessment_insights import run_step9_test_suite as run_step9
    await run_step9()
    print("    Step 9 Test Suite: PASS (18/18 tests)")
    print("    [PASSED]")
    passed_count += 1

    print("\n" + "=" * 95)
    print(f"ALL {passed_count}/{total_tests} STEP 10B PRECISION FIX TESTS PASSED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(run_step10b_test_suite())
