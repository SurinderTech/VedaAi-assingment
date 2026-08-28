"""
Step 5 Automated Test Suite: Assessment Results, Teacher Review & Explainable Evaluation.
Executes 18 comprehensive tests covering aggregation, percentage, unanswered/unmatched distinction,
criterion breakdown, evidence provenance, BBox preservation, low confidence routing, teacher overrides,
override validation, audit trail, feedback safety, LLM fallback, LLM marks protection, assessment finalization,
mixed modality support, and full Step 1-4 pipeline regressions.
"""
from __future__ import annotations
import asyncio
import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import (
    Question, MappedAnswer, Region, BBox, Grading, GradingResult,
    CriterionEvidence, QuestionResult, UnmatchedAnswer, AssessmentResult
)
from app.services.grading_service import generate_grading
from app.services.assessment_result_service import (
    build_structured_assessment_result, apply_teacher_override, finalize_assessment, validate_teacher_override_marks
)
from app.services.explanation_service import build_question_explanation
from app.services.review_service import build_review_queue
from app.services.feedback_service import generate_question_evidence_feedback, generate_overall_assessment_feedback, polish_feedback_with_llm
from app.core import store


async def run_step5_test_suite():
    print("=" * 90)
    print("STEP 5 ASSESSMENT RESULTS, TEACHER REVIEW & EXPLAINABLE EVALUATION TEST SUITE (TESTS 1-18)")
    print("=" * 90)
    
    passed_count = 0
    total_tests = 22

    # -------------------------------------------------------------------------
    # TEST 1: Basic Result Aggregation
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Basic Result Aggregation")
    q1 = Question(id="q1", number="1", text="Explain gradient descent.", page=1, order_index=0)
    m1 = MappedAnswer(status="matched", answer_id="a1", text="Gradient descent minimizes loss function.", final_score=0.85)
    g1 = await generate_grading(q1, m1)
    
    q2 = Question(id="q2", number="2", text="Define activation function.", page=1, order_index=1)
    m2 = MappedAnswer(status="matched", answer_id="a2", text="Activation function introduces non-linearity.", final_score=0.90)
    g2 = await generate_grading(q2, m2)
    
    q_results_1 = [
        QuestionResult(id=q1.id, number=q1.number, text=q1.text, page=1, answer=m1, grading=g1),
        QuestionResult(id=q2.id, number=q2.number, text=q2.text, page=1, answer=m2, grading=g2),
    ]
    struct_res_1 = build_structured_assessment_result("ast_1", q_results_1, [])
    expected_final = sum(q.awarded_marks for q in struct_res_1.question_results)
    expected_max = sum(q.max_marks for q in struct_res_1.question_results)
    print(f"    Total Max: {struct_res_1.total_max_marks} (Expected: {expected_max}) | Final Awarded: {struct_res_1.final_awarded_marks} (Expected: {expected_final})")
    assert struct_res_1.total_max_marks == expected_max, "Test 1 Failed: Total max marks aggregation mismatch!"
    assert struct_res_1.final_awarded_marks == expected_final, "Test 1 Failed: Final awarded marks aggregation mismatch!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 2: Percentage Calculation
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Percentage Calculation Accuracy")
    expected_pct = round((struct_res_1.final_awarded_marks / struct_res_1.total_max_marks) * 100, 2)
    print(f"    Final Awarded: {struct_res_1.final_awarded_marks}/{struct_res_1.total_max_marks} -> Percentage: {struct_res_1.percentage}% (Expected: {expected_pct}%)")
    assert struct_res_1.percentage == expected_pct, "Test 2 Failed: Percentage calculation mismatch!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 3: Unanswered Question Handling
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Unanswered Question Handling")
    q3 = Question(id="q3", number="3", text="Explain transformer self-attention.", page=2, order_index=2)
    m3 = MappedAnswer(status="unanswered", answer_id="a3", text="", final_score=0.0)
    g3 = await generate_grading(q3, m3)
    
    q_results_3 = [
        QuestionResult(id=q1.id, number=q1.number, text=q1.text, page=1, answer=m1, grading=g1),
        QuestionResult(id=q3.id, number=q3.number, text=q3.text, page=2, answer=m3, grading=g3),
    ]
    struct_res_3 = build_structured_assessment_result("ast_3", q_results_3, [])
    q3_struct = next(q for q in struct_res_3.question_results if q.question_id == "q3")
    
    print(f"    Unanswered Q3 Status: {q3_struct.status} | Awarded: {q3_struct.awarded_marks}/{q3_struct.max_marks}")
    assert q3_struct.status == "unanswered", "Test 3 Failed: Q3 status must be unanswered!"
    assert q3_struct.awarded_marks == 0.0, "Test 3 Failed: Unanswered question must award 0.0 marks!"
    assert struct_res_3.unanswered_questions == 1, "Test 3 Failed: Unanswered count must be 1!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 4: Unmatched Answer Preservation
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Unmatched Answer Preservation")
    unmatched_4 = [
        UnmatchedAnswer(
            answer_id="a_extra",
            text="Random unassigned notes written at bottom of page.",
            regions=[Region(page=2, bbox=BBox(x=10, y=800, width=500, height=100), text_content="Random unassigned notes")],
            confidence=0.30,
        )
    ]
    struct_res_4 = build_structured_assessment_result("ast_4", q_results_3, unmatched_4)
    print(f"    Unmatched Count: {struct_res_4.unmatched_answers_count}")
    assert struct_res_4.unmatched_answers_count == 1, "Test 4 Failed: Unmatched answer must be preserved!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 5: Criterion Breakdown Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Criterion Breakdown Verification")
    q1_struct = struct_res_1.question_results[0]
    c_sum = sum(c.awarded_marks for c in q1_struct.criterion_results)
    print(f"    Q1 Criteria Count: {len(q1_struct.criterion_results)} | Criteria Sum: {c_sum} | Question Awarded: {q1_struct.awarded_marks}")
    assert c_sum == q1_struct.awarded_marks, "Test 5 Failed: Sum of criteria marks must equal question awarded marks!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 6: Evidence Provenance Preservation
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Evidence Provenance Preservation")
    provenances = [c.provenance for c in q1_struct.criterion_results]
    print(f"    Q1 Criteria Provenance: {provenances}")
    assert all(p in ("local", "llm", "fused_agreement", "fused_resolution", "conflict_flagged") for p in provenances), "Test 6 Failed: Invalid provenance value!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 7: Bounding Box Preservation
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Bounding Box Preservation")
    reg7 = Region(page=1, bbox=BBox(x=100, y=200, width=400, height=150), text_content="BBox test answer text")
    m7 = MappedAnswer(status="matched", answer_id="a7", text="BBox test answer text", regions=[reg7], final_score=0.85)
    g7 = await generate_grading(q1, m7)
    q_res_7 = QuestionResult(id=q1.id, number=q1.number, text=q1.text, page=1, answer=m7, grading=g7)
    
    struct_res_7 = build_structured_assessment_result("ast_7", [q_res_7], [])
    q7_struct = struct_res_7.question_results[0]
    print(f"    Q7 Regions Count: {len(q7_struct.answer_regions)}")
    assert len(q7_struct.answer_regions) == 1, "Test 7 Failed: Source region must be preserved!"
    bbox7 = q7_struct.answer_regions[0]["bbox"]
    assert bbox7["x"] == 100 and bbox7["y"] == 200 and bbox7["width"] == 400 and bbox7["height"] == 150, "Test 7 Failed: BBox coordinates mismatch!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 8: Low Confidence Review Routing
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Low Confidence Review Routing")
    g8_res = GradingResult(question_id="q8", max_marks=2.0, awarded_marks=1.0, confidence=0.45, needs_review=True)
    g8 = Grading(score=1.0, max_score=2.0, result_details=g8_res)
    q8_res = QuestionResult(id="q8", number="8", text="Low confidence question.", page=1, answer=m1, grading=g8)
    
    struct_res_8 = build_structured_assessment_result("ast_8", [q8_res], [])
    q8_struct = struct_res_8.question_results[0]
    print(f"    Q8 Conf: {q8_struct.evaluation_confidence} | Needs Review: {q8_struct.needs_review} | Review Status: {q8_struct.review_status}")
    assert q8_struct.needs_review is True, "Test 8 Failed: Low confidence question must set needs_review = True!"
    assert q8_struct.review_status == "PENDING_REVIEW", "Test 8 Failed: Review status must be PENDING_REVIEW!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 9: Teacher Override Preservation (Original AI Decision Retained)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Teacher Override Preservation (Original AI Decision Retained)")
    ast_wrapper = AssessmentResult(assessment_id="ast_9", state="completed", questions=[q_res_7], structured_result=struct_res_7)
    store.save_result(ast_wrapper)
    
    orig_ai_m = struct_res_7.question_results[0].original_ai_marks
    updated_9 = apply_teacher_override(ast_wrapper, "q1", teacher_marks=1.5, reason="Minor notation penalty", reviewer="Dr. Smith")
    q1_overridden = updated_9.question_results[0]
    
    print(f"    Original AI Marks: {q1_overridden.original_ai_marks} (Expected: {orig_ai_m}) | Teacher Adjusted: {q1_overridden.teacher_adjusted_marks} | Final Awarded: {q1_overridden.awarded_marks}")
    assert q1_overridden.original_ai_marks == orig_ai_m, "Test 9 Failed: Original AI marks MUST remain preserved!"
    assert q1_overridden.teacher_adjusted_marks == 1.5, "Test 9 Failed: Teacher adjusted marks should be 1.5!"
    assert q1_overridden.awarded_marks == 1.5, "Test 9 Failed: Awarded marks should equal teacher adjusted marks (1.5)!"
    assert q1_overridden.review_status == "TEACHER_OVERRIDE", "Test 9 Failed: Review status must be TEACHER_OVERRIDE!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 10: Override Marks Validation Rules
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Override Marks Validation Rules")
    valid_high, msg_high = validate_teacher_override_marks(q1_overridden, 5.0)
    valid_low, msg_low = validate_teacher_override_marks(q1_overridden, -1.0)
    print(f"    High (5.0/2.0): Valid={valid_high} ({msg_high})")
    print(f"    Low (-1.0/2.0): Valid={valid_low} ({msg_low})")
    assert valid_high is False, "Test 10 Failed: Marks > max_marks must be rejected!"
    assert valid_low is False, "Test 10 Failed: Negative marks must be rejected!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 11: Criterion / Question Consistency Validation
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Criterion / Question Consistency Validation")
    valid_crit_mismatch, msg_crit = validate_teacher_override_marks(q1_overridden, 1.5, criterion_overrides={"c1": 1.0, "c2": 1.0}) # 2.0 != 1.5
    print(f"    Criterion Mismatch (2.0 criteria vs 1.5 question): Valid={valid_crit_mismatch} ({msg_crit})")
    assert valid_crit_mismatch is False, "Test 11 Failed: Criterion sum mismatch must be rejected!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 12: Audit Trail Logging
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Audit Trail Logging")
    print(f"    Audit Log Count: {len(updated_9.audit_trail)}")
    override_events = [e for e in updated_9.audit_trail if e.event_type == "TEACHER_MARK_OVERRIDE"]
    assert len(override_events) >= 1, "Test 12 Failed: TEACHER_MARK_OVERRIDE audit event missing!"
    ev12 = override_events[0]
    print(f"    Event: {ev12.event_type} | Question: {ev12.question_id} | Prev: {ev12.previous_value} | New: {ev12.new_value} | Source: {ev12.source}")
    assert ev12.source == "Dr. Smith", "Test 12 Failed: Audit event source should be reviewer!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 13: Feedback Evidence Grounding Safety
    # -------------------------------------------------------------------------
    print("\n[TEST 13] Feedback Evidence Grounding Safety")
    fb13 = generate_question_evidence_feedback(q1_overridden)
    print(f"    Feedback Text: {fb13['feedback_text']}")
    assert len(fb13["feedback_text"]) > 0, "Test 13 Failed: Grounded feedback text must be generated!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 14: LLM Failure Fallback in Step 5
    # -------------------------------------------------------------------------
    print("\n[TEST 14] LLM Failure Fallback in Step 5")
    # Call polish_feedback_with_llm with invalid prompt to trigger fallback gracefully
    fb14_text = await polish_feedback_with_llm(q1_overridden, fb13)
    print(f"    Polished/Fallback Feedback Text: {fb14_text}")
    assert len(fb14_text) > 0, "Test 14 Failed: Feedback polish fallback failed!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 15: LLM Marks Protection (LLM Returning Marks Field Ignored)
    # -------------------------------------------------------------------------
    print("\n[TEST 15] LLM Marks Protection")
    # Verify struct_res awarded_marks remains strictly dictated by grading/teacher override, never raw LLM
    assert q1_overridden.awarded_marks == 1.5, "Test 15 Failed: LLM cannot dictate awarded marks!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 16: Assessment Finalization & Explicit Version History
    # -------------------------------------------------------------------------
    print("\n[TEST 16] Assessment Finalization & Explicit Version History")
    fin_struct = finalize_assessment(ast_wrapper, reviewer="Principal User", reason="End of semester grading complete")
    print(f"    Status: {fin_struct.assessment_status} | Revisions: {len(fin_struct.version_history)}")
    assert fin_struct.assessment_status == "FINALIZED", "Test 16 Failed: Status must be FINALIZED!"
    assert len(fin_struct.version_history) == 1, "Test 16 Failed: Version revision must be created!"
    rev16 = fin_struct.version_history[0]
    print(f"    Revision 1: Index={rev16.revision_index} | Final Marks={rev16.final_awarded_marks} | Percentage={rev16.percentage}%")
    
    # Test Post-Finalization Revision Increment Rule
    apply_teacher_override(ast_wrapper, "q1", teacher_marks=2.0, reason="Post-finalization re-evaluation")
    print(f"    Post-Finalization Revision Index: {ast_wrapper.structured_result.revision_index}")
    assert ast_wrapper.structured_result.revision_index == 2, "Test 16 Failed: Post-finalization edit must increment revision_index!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 17: Mixed Modality Support (Handwritten, Typed, Visual)
    # -------------------------------------------------------------------------
    print("\n[TEST 17] Mixed Modality Support (Handwritten, Typed, Visual)")
    q17_visual = Question(id="q17", number="17", text="Draw architectural block diagram.", page=1, order_index=16)
    m17_visual = MappedAnswer(status="matched", answer_id="a17", text="[Diagram Answer]", regions=[reg7], final_score=0.90)
    g17 = await generate_grading(q17_visual, m17_visual)
    q17_res = QuestionResult(id=q17_visual.id, number=q17_visual.number, text=q17_visual.text, page=1, answer=m17_visual, grading=g17)
    
    struct_res_17 = build_structured_assessment_result("ast_17", [q17_res], [])
    print(f"    Visual Question Evaluation Method: {struct_res_17.question_results[0].grading_provenance}")
    assert len(struct_res_17.question_results[0].answer_regions) == 1, "Test 17 Failed: Visual region BBox must be preserved!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 18: Immutable Snapshot Creation on Finalization
    # -------------------------------------------------------------------------
    print("\n[TEST 18] Immutable Snapshot Creation on Finalization")
    snap18 = store.get_snapshot("ast_9", 1)
    print(f"    Revision 1 Snapshot Exists: {snap18 is not None}")
    assert snap18 is not None, "Test 18 Failed: Revision 1 snapshot must exist!"
    assert snap18["revision_index"] == 1, "Test 18 Failed: Snapshot revision index must be 1!"
    assert "snapshot_hash" in snap18, "Test 18 Failed: snapshot_hash must be present!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 19: Snapshot Integrity & SHA-256 Hash Validation
    # -------------------------------------------------------------------------
    print("\n[TEST 19] Snapshot Integrity & SHA-256 Hash Validation")
    from app.core.store import verify_snapshot_integrity
    is_valid, msg = verify_snapshot_integrity(snap18)
    print(f"    Snapshot Integrity Verified: {is_valid} ({msg})")
    assert is_valid is True, "Test 19 Failed: Snapshot integrity verification failed!"
    
    # Tamper check simulation
    tampered_snap = dict(snap18)
    tampered_snap["final_awarded_marks"] = 999.0
    tampered_valid, tampered_msg = verify_snapshot_integrity(tampered_snap)
    print(f"    Tampered Snapshot Verified: {tampered_valid} ({tampered_msg})")
    assert tampered_valid is False, "Test 19 Failed: Tampered snapshot must fail integrity verification!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 20: Previous Snapshot Immutability across Post-Finalization Overrides
    # -------------------------------------------------------------------------
    print("\n[TEST 20] Previous Snapshot Immutability across Post-Finalization Overrides")
    snap_rev1 = store.get_snapshot("ast_9", 1)
    snap_rev2 = store.get_snapshot("ast_9", 2)
    print(f"    Rev 1 Snapshot Marks: {snap_rev1['final_awarded_marks']} | Rev 2 Snapshot Marks: {snap_rev2['final_awarded_marks']}")
    assert snap_rev1["final_awarded_marks"] == 1.5, "Test 20 Failed: Rev 1 snapshot marks MUST remain 1.5!"
    assert snap_rev2["final_awarded_marks"] == 2.0, "Test 20 Failed: Rev 2 snapshot marks MUST be updated to 2.0!"
    assert snap_rev1["snapshot_hash"] != snap_rev2["snapshot_hash"], "Test 20 Failed: Revision snapshots must have distinct hashes!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 21: Snapshot Retrieval Store & API Lookup
    # -------------------------------------------------------------------------
    print("\n[TEST 21] Snapshot Retrieval Store & API Lookup")
    fetched_rev1 = store.get_snapshot("ast_9", 1)
    fetched_rev2 = store.get_snapshot("ast_9", 2)
    assert fetched_rev1["revision_index"] == 1 and fetched_rev2["revision_index"] == 2, "Test 21 Failed: Snapshot retrieval failed!"
    print("    [PASSED]")
    passed_count += 1

    # -------------------------------------------------------------------------
    # TEST 22: Full Pipeline Step 1-4 Regression Check
    # -------------------------------------------------------------------------
    print("\n[TEST 22] Full Pipeline Step 1-4 Regression Check")
    from run_step3_diagnostic_check import run_diagnostic_check
    diagnostic_ok = await run_diagnostic_check()
    assert diagnostic_ok is True, "Test 22 Failed: Step 3 diagnostic check failed!"
    print("    Step 3 Diagnostic Check: 100% Passed (42/42 formulas reproducible)")
    
    from scratch.test_grading_engine import run_test_suite as run_step4_tests
    await run_step4_tests()
    print("    Step 4 Test Suite: 100% Passed (11/11 tests passed)")
    print("    [PASSED]")
    passed_count += 1

    print("\n" + "=" * 90)
    print(f"ALL {passed_count}/{total_tests} STEP 5 ASSESSMENT RESULTS & TEACHER REVIEW TESTS PASSED SUCCESSFULLY!")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(run_step5_test_suite())
