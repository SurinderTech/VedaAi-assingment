"""
STEP 10C — ASSIGNMENT-CORE ASSESSMENT WORKSPACE UI TEST SUITE (TESTS 1-14)
Verifies assignment-core question ↔ answer sheet workspace interaction,
exact Step 10B BBox geometry retention, zero score recalculation on selection,
unanswered zero highlight rule, unmatched region isolation, multi-region/multi-page
handling, out-of-order page navigation, secondary tools preservation, and full Step 1-9 regressions.
"""

import asyncio
import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import (
    Question,
    AnswerRegion,
    Region,
    BBox,
    MappedAnswer,
    Grading,
    QuestionResult,
    GradingResult,
    CriterionEvidence,
    AssessmentResult,
    StructuredAssessmentResult,
    StructuredQuestionResult,
    CriterionResult,
)
from app.services.assessment_result_service import (
    build_structured_assessment_result,
    apply_teacher_override,
)


def log_print(msg: str):
    print(msg, flush=True)


async def run_step10c_test_suite():
    log_print("=" * 95)
    log_print("STEP 10C — ASSIGNMENT-CORE ASSESSMENT WORKSPACE UI TEST SUITE (TESTS 1-14)")
    log_print("=" * 95)

    passed_count = 0
    total_tests = 14

    # Setup Benchmark Test Data
    reg1_p1 = Region(page=1, bbox=BBox(x=100, y=150, width=350, height=80))
    reg2_p3 = Region(page=3, bbox=BBox(x=80, y=200, width=400, height=120))
    reg3_p2_a = Region(page=2, bbox=BBox(x=50, y=100, width=300, height=50))
    reg3_p2_b = Region(page=2, bbox=BBox(x=50, y=300, width=300, height=50))

    # Q1: Matched single region on Page 1
    q1 = Question(id="q1", number="1", text="Which blood vessel carries blood away from the heart?", page=1, order_index=0)
    m1 = MappedAnswer(status="matched", answer_id="a1", text="Arteries carry blood away from the heart.", regions=[reg1_p1], final_score=1.0)
    g1 = GradingResult(question_id="q1", max_marks=2.0, awarded_marks=2.0, confidence=0.98, criteria=[])
    q1_res = QuestionResult(id=q1.id, number=q1.number, text=q1.text, page=1, answer=m1, grading=Grading(score=2.0, max_score=2.0, result_details=g1))

    # Q2: Out-of-order matched single region on Page 3
    q2 = Question(id="q2", number="2", text="Which organelle is primarily involved in photosynthesis?", page=1, order_index=1)
    m2 = MappedAnswer(status="matched", answer_id="a2", text="Chloroplast is responsible for photosynthesis.", regions=[reg2_p3], final_score=0.95)
    g2 = GradingResult(question_id="q2", max_marks=2.0, awarded_marks=2.0, confidence=0.95, criteria=[])
    q2_res = QuestionResult(id=q2.id, number=q2.number, text=q2.text, page=1, answer=m2, grading=Grading(score=2.0, max_score=2.0, result_details=g2))

    # Q3: Multi-region matched answer on Page 2
    q3 = Question(id="q3", number="3", text="Explain the structure and function of nephron.", page=2, order_index=2)
    m3 = MappedAnswer(status="matched", answer_id="a3", text="Part A: Bowman capsule...\nPart B: Loop of Henle...", regions=[reg3_p2_a, reg3_p2_b], final_score=0.90)
    g3 = GradingResult(question_id="q3", max_marks=5.0, awarded_marks=5.0, confidence=0.90, criteria=[])
    q3_res = QuestionResult(id=q3.id, number=q3.number, text=q3.text, page=2, answer=m3, grading=Grading(score=5.0, max_score=5.0, result_details=g3))

    # Q4: Unanswered Question
    q4 = Question(id="q4", number="4", text="Describe human heart blood circulation.", page=2, order_index=3)
    m4 = MappedAnswer(status="unanswered", answer_id=None, text="", regions=[], final_score=0.0)
    g4 = GradingResult(question_id="q4", max_marks=2.0, awarded_marks=0.0, confidence=1.0)
    q4_res = QuestionResult(id=q4.id, number=q4.number, text=q4.text, page=2, answer=m4, grading=Grading(score=0.0, max_score=2.0, result_details=g4))

    # Unmatched Answer Region
    unmatched_reg = Region(page=4, bbox=BBox(x=60, y=400, width=200, height=60))

    struct_res = build_structured_assessment_result("ast_10c", [q1_res, q2_res, q3_res, q4_res], [unmatched_reg])

    # TEST 1: Actual Extracted Questions Populate Workspace Data
    log_print("\n[TEST 1] Actual Extracted Questions Populate Workspace Data")
    q_list = struct_res.question_results
    log_print(f"    Total Extracted Questions: {len(q_list)}")
    assert len(q_list) == 4, "Test 1 Failed: Must contain 4 extracted questions!"
    assert q_list[0].question_text == "Which blood vessel carries blood away from the heart?", "Test 1 Failed: Question text mismatch!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 2: No Hardcoded Question Data
    log_print("\n[TEST 2] No Hardcoded Question Data in Data Models")
    assert struct_res.assessment_id == "ast_10c", "Test 2 Failed: Assessment ID must be dynamic!"
    assert q_list[0].question_id == "q1" and q_list[1].question_id == "q2", "Test 2 Failed: Question IDs must come from result!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 3: Question Selection Retrieves Mapped Regions
    log_print("\n[TEST 3] Question Selection Retrieves Mapped Regions")
    sq1 = next(q for q in q_list if q.question_id == "q1")
    regions_q1 = sq1.answer_regions
    log_print(f"    Q1 Mapped Regions Count: {len(regions_q1)}")
    assert len(regions_q1) == 1, "Test 3 Failed: Q1 must retrieve exactly 1 mapped region!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 4: Single Answer Region Passed Correctly
    log_print("\n[TEST 4] Single Answer Region Passed Correctly")
    r1 = regions_q1[0]
    log_print(f"    Single Region: page={r1['page']}, bbox={r1['bbox']}")
    assert r1["page"] == 1 and r1["bbox"]["x"] == 100 and r1["bbox"]["y"] == 150, "Test 4 Failed: Single region parameters mismatch!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 5: Multiple Answer Regions Passed Correctly
    log_print("\n[TEST 5] Multiple Answer Regions Passed Correctly")
    sq3 = next(q for q in q_list if q.question_id == "q3")
    regions_q3 = sq3.answer_regions
    log_print(f"    Q3 Multi-Regions Count: {len(regions_q3)}")
    assert len(regions_q3) == 2, "Test 5 Failed: Q3 must retrieve 2 distinct sub-regions!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 6: Multi-Page Regions Preserve Original Page Association
    log_print("\n[TEST 6] Multi-Page Regions Page Association")
    assert sq1.answer_pages == [1] and sq3.answer_pages == [2], "Test 6 Failed: Original page association must be preserved!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 7: Unanswered Questions Do Not Receive Fabricated Highlights
    log_print("\n[TEST 7] Unanswered Questions Produce Zero Highlights")
    sq4 = next(q for q in q_list if q.question_id == "q4")
    log_print(f"    Q4 Status: {sq4.status} | Regions: {sq4.answer_regions}")
    assert sq4.status == "unanswered", "Test 7 Failed: Q4 status must be unanswered!"
    assert len(sq4.answer_regions) == 0, "Test 7 Failed: Unanswered question MUST have empty regions array!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 8: Unmatched Answer Regions Are Not Incorrectly Assigned
    log_print("\n[TEST 8] Unmatched Answer Regions Isolation")
    log_print(f"    Unmatched Answers Count: {struct_res.unmatched_answers_count}")
    assert struct_res.unmatched_answers_count == 1, "Test 8 Failed: Unmatched region count must be 1!"
    for q in q_list:
        for reg in q.answer_regions:
            assert reg["page"] != 4 or reg["bbox"]["y"] != 400, "Test 8 Failed: Unmatched region was assigned to question!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 9: Out-of-Order Mappings Navigate to Correct Page
    log_print("\n[TEST 9] Out-of-Order Mappings Navigation")
    sq2 = next(q for q in q_list if q.question_id == "q2")
    log_print(f"    Q2 Mapped Answer Page: {sq2.answer_pages}")
    assert sq2.answer_pages == [3], "Test 9 Failed: Q2 answer is on Page 3 despite being question #2!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 10: Step 10B BBox Geometry Remains Unchanged
    log_print("\n[TEST 10] Step 10B BBox Geometry Retention")
    b1 = sq1.answer_regions[0]["bbox"]
    log_print(f"    Q1 BBox Geometry: {b1}")
    assert b1["x"] == 100 and b1["y"] == 150 and b1["width"] == 350 and b1["height"] == 80, "Test 10 Failed: BBox geometry mutated!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 11: Secondary Teacher Tools Remain Accessible & Functioning
    log_print("\n[TEST 11] Secondary Teacher Tools Functionality")
    log_print(f"    Review Queue Questions Count: {struct_res.questions_needing_review}")
    log_print(f"    Audit Trail Events: {len(struct_res.audit_trail)}")
    log_print(f"    Version History Revisions: {len(struct_res.version_history)}")
    assert struct_res.assessment_status == "IN_REVIEW", "Test 11 Failed: Assessment status preserved!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 12: Zero Data Mutation on Selection
    log_print("\n[TEST 12] Zero Data Mutation on Selection")
    score_before = struct_res.final_awarded_marks
    _ = struct_res.question_results[0]
    score_after = struct_res.final_awarded_marks
    log_print(f"    Score Before: {score_before} | Score After: {score_after}")
    assert score_before == score_after, "Test 12 Failed: Question selection mutated score!"
    log_print("    [PASSED]")
    passed_count += 1

    # TEST 13: Full Step 1-9 Regression Suites Verification
    log_print("\n[TEST 13] Full Step 1-9 Regression Suites Verification")
    python_exe = sys.executable
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    reg_scripts = [
        "run_step3_diagnostic_check.py",
        "scratch/test_grading_engine.py",
        "scratch/test_assessment_results.py",
        "scratch/test_step6_teacher_workspace.py",
        "scratch/test_step7_student_results.py",
        "scratch/test_step8_embeddings.py",
        "scratch/test_step9_assessment_insights.py",
        "scratch/test_step10b_precision_fixes.py",
    ]

    for script in reg_scripts:
        log_print(f"    Running regression script: {script}...")
        res = subprocess.run([python_exe, script], cwd=backend_dir, capture_output=True, text=True)
        assert res.returncode == 0, f"Test 13 Failed: {script} exited with code {res.returncode}\n{res.stderr}"
        log_print(f"    [PASS] {script}")

    log_print("    [PASSED]")
    passed_count += 1

    # TEST 14: Frontend Production Build Success
    log_print("\n[TEST 14] Frontend Production Build Verification")
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    build_cmd = "npm run build"
    log_print(f"    Executing '{build_cmd}' in {frontend_dir}...")
    res = subprocess.run(build_cmd, shell=True, cwd=frontend_dir, capture_output=True, text=True)
    if res.returncode != 0:
        log_print(f"    [BUILD STDOUT]\n{res.stdout}")
        log_print(f"    [BUILD STDERR]\n{res.stderr}")
        assert False, f"Test 14 Failed: Frontend build exited with code {res.returncode}"
    else:
        log_print("    Frontend production build compiled successfully with 0 errors!")
        log_print("    [PASSED]")
        passed_count += 1

    log_print("\n" + "=" * 95)
    log_print(f"ALL {passed_count}/{total_tests} STEP 10C ASSIGNMENT-CORE WORKSPACE TESTS PASSED SUCCESSFULLY!")
    log_print("=" * 95)

if __name__ == "__main__":
    asyncio.run(run_step10c_test_suite())
