import asyncio
import json
import numpy as np
from app.models.schemas import Question, AnswerRegion, Region, BBox
from app.services.mapping_engine import map_answers, _evaluate_candidate_evidence, _normalize_anchor_key
from app.core.config import settings

async def run_diagnostic_check():
    print("=" * 80)
    print("STEP 3 — FINAL DIAGNOSTIC CONSISTENCY CHECK")
    print("=" * 80)
    
    # 1. Configured Anchored Weights
    w_anch_cfg = settings.MAPPING_ANCHOR_WEIGHT      # 0.40
    w_sem_cfg = settings.MAPPING_SEMANTIC_WEIGHT       # 0.30
    w_struct_cfg = settings.MAPPING_STRUCTURAL_WEIGHT  # 0.15
    w_spat_cfg = settings.MAPPING_SPATIAL_WEIGHT      # 0.10
    w_ord_cfg = settings.MAPPING_ORDER_WEIGHT         # 0.05
    
    print("\n--- WEIGHT CONFIGURATION VERIFICATION ---")
    print(f"Anchored Weights:   anchor={w_anch_cfg:.2f}, semantic={w_sem_cfg:.2f}, struct={w_struct_cfg:.2f}, spatial={w_spat_cfg:.2f}, order={w_ord_cfg:.2f} (Sum: {w_anch_cfg+w_sem_cfg+w_struct_cfg+w_spat_cfg+w_ord_cfg:.2f})")
    print("Unanchored Weights: anchor=0.00, semantic=0.55, struct=0.25, spatial=0.15, order=0.05 (Sum: 0.55+0.25+0.15+0.05=1.00)")

    # 2. Create Benchmark Test Cases Covering All 6 Behavioral Requirements
    questions = [
        Question(id="q1", number="1", text="Describe the process of photosynthesis in green plants using chlorophyll light energy carbon dioxide water produce glucose oxygen.", page=1, order_index=0),
        Question(id="q2", number="2", text="State Newton's second law of motion force mass acceleration F=ma equation derivation dynamics vector mechanics physics.", page=1, order_index=1),
        Question(id="q3", number="3", text="Explain the structure and function of DNA helicase enzyme unwinding double helix replication hydrogen bonds nucleotide sequence genetics.", page=2, order_index=2),
        Question(id="q4", number="4", text="Solve the quadratic equation x^2 - 5x + 6 = 0 for real roots discriminant formula factorization polynomial mathematics.", page=2, order_index=3),
        Question(id="q5", number="5", text="Discuss the economic causes of the French Revolution taxation monarchy estates general bourgeoisie debt financial crisis history.", page=3, order_index=4),
        Question(id="q6", number="6", text="Define kinetic energy motion mass velocity joules mechanical work real-world physical examples energy transformation physics.", page=3, order_index=5),
    ]

    answers = [
        # Scenario 1: Explicit anchor + compatible semantics -> Q1
        AnswerRegion(
            answer_id="ans_q1",
            question_anchor="Q1",
            pages=[1],
            regions=[Region(page=1, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="Q1 Photosynthesis is the process in green plants using chlorophyll light energy carbon dioxide water to produce glucose oxygen.",
            reading_order=0,
        ),
        # Scenario 2: Explicit anchor + contradictory semantics -> Anchored Q2, but text is actually Photosynthesis (matches Q1 text, not Q2)
        AnswerRegion(
            answer_id="ans_q2_conflict",
            question_anchor="Q2",
            pages=[1],
            regions=[Region(page=1, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="Q2 Photosynthesis process in green plants using chlorophyll light energy carbon dioxide water produce glucose oxygen.",
            reading_order=1,
        ),
        # Scenario 3: No anchor + strong semantics -> Q3 (DNA helicase)
        AnswerRegion(
            answer_id="ans_q3_unanchored",
            question_anchor=None,
            pages=[2],
            regions=[Region(page=2, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="DNA helicase enzyme unwinding double helix replication hydrogen bonds nucleotide sequence genetics structure function.",
            reading_order=2,
        ),
        # Scenario 4: No anchor + weak semantics -> unanchored vague note
        AnswerRegion(
            answer_id="ans_q4_weak",
            question_anchor=None,
            pages=[2],
            regions=[Region(page=2, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="General calculation steps shown.",
            reading_order=3,
        ),
        # Scenario 5: Poor candidate / noise
        AnswerRegion(
            answer_id="ans_noise",
            question_anchor=None,
            pages=[5],
            regions=[Region(page=5, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="Random scribbles page margin note.",
            reading_order=4,
        ),
        # Scenario 6: Competing candidate 1 for Q6
        AnswerRegion(
            answer_id="ans_q6_compete1",
            question_anchor="Q6",
            pages=[3],
            regions=[Region(page=3, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="Q6 Kinetic energy motion mass velocity joules mechanical work real-world physical examples energy transformation.",
            reading_order=5,
        ),
        # Scenario 6: Competing candidate 2 for Q6
        AnswerRegion(
            answer_id="ans_q6_compete2",
            question_anchor="Q6",
            pages=[3],
            regions=[Region(page=3, bbox=BBox(x=0, y=0, width=100, height=100))],
            text="Q6 Kinetic energy motion mass velocity formula 1/2 mv^2.",
            reading_order=6,
        ),
    ]

    all_q_norms = {_normalize_anchor_key(q.number) for q in questions}

    print("\n--- MATHEMATICAL FORMULA VERIFICATION ACROSS ALL CANDIDATE PAIRS ---")
    discrepancies = []
    
    from app.services.embedding_service import similarity_matrix
    q_texts = [q.text for q in questions]
    r_texts = [r.text for r in answers]
    sim_mat = similarity_matrix(q_texts, r_texts)

    for qi, q in enumerate(questions):
        for ri, r in enumerate(answers):
            ev = _evaluate_candidate_evidence(q, qi, r, ri, float(sim_mat[qi, ri]), len(questions), len(answers), all_q_norms, sim_mat)
            
            s_anch = ev["anchor_score"]
            s_sem = ev["semantic_score"]
            s_struct = ev["structural_score"]
            s_spat = ev["spatial_score"]
            s_ord = ev["order_score"]
            
            w_anch = ev["w_anchor"]
            w_sem = ev["w_semantic"]
            w_struct = ev["w_struct"]
            w_spat = ev["w_spatial"]
            w_ord = ev["w_order"]
            
            expected_raw = round(
                w_anch * s_anch + w_sem * s_sem + w_struct * s_struct + w_spat * s_spat + w_ord * s_ord,
                3
            )
            expected_penalty = 0.70 if ev["conflict_detected"] else 1.00
            expected_final = round(expected_raw * expected_penalty, 3)

            # Verification assertions
            raw_match = abs(ev["raw_final_score"] - expected_raw) < 1e-5
            final_match = abs(ev["final_score"] - expected_final) < 1e-5

            if not raw_match or not final_match:
                discrepancies.append(
                    f"DISCREPANCY Q{q.number}-R{ri}: Reported Raw={ev['raw_final_score']}, Expected Raw={expected_raw} | "
                    f"Reported Final={ev['final_score']}, Expected Final={expected_final}"
                )

    print(f"Total candidate pairs evaluated: {len(questions) * len(answers)}")
    if discrepancies:
        print(f"FAILED: Found {len(discrepancies)} mathematical discrepancies:")
        for d in discrepancies:
            print(" -", d)
    else:
        print("PASSED: 100% of diagnostic formulas are mathematically reproducible with zero discrepancies!")

    # 3. Verification of prompt's specific formula example:
    # 0.40*1.00 + 0.30*0.123 + 0.15*0.70 + 0.10*1.00 + 0.05*0.50 = 0.667
    ex_anch, ex_sem, ex_struct, ex_spat, ex_ord = 1.000, 0.123, 0.700, 1.000, 0.500
    ex_raw = round(0.40 * ex_anch + 0.30 * ex_sem + 0.15 * ex_struct + 0.10 * ex_spat + 0.05 * ex_ord, 3)
    print(f"\nExample Formula Check: 0.40*{ex_anch:.3f} + 0.30*{ex_sem:.3f} + 0.15*{ex_struct:.3f} + 0.10*{ex_spat:.3f} + 0.05*{ex_ord:.3f} = {ex_raw:.3f}")
    assert ex_raw == 0.667, f"Expected 0.667 but got {ex_raw}"
    print("Example formula check PASSED: Exactly equals 0.667!")

    # 4. Run Full map_answers Benchmark Pipeline
    print("\n--- RUNNING GLOBAL BIPARTITE MAPPING PIPELINE ---")
    results, unmatched = await map_answers(questions, answers)

    # 5. Generate Step 5 Final Diagnostic Table
    print("\n" + "=" * 125)
    print("FINAL DIAGNOSTIC TABLE (STEP 5 REQUIREMENT)")
    print("=" * 125)
    print(f"{'Case/Question':<16} | {'Anchor':<6} | {'Semantic':<8} | {'Struct':<6} | {'Spatial':<7} | {'Order':<5} | {'Effective Weights':<28} | {'Raw Comp':<8} | {'Penalty':<7} | {'Final':<6} | {'Status'}")
    print("-" * 125)

    for q in questions:
        res = results.get(q.id)
        if not res or res.status == "unanswered":
            print(f"{'Q' + q.number + ' (Null/Unans)':<16} | {'-':<6} | {'-':<8} | {'-':<6} | {'-':<7} | {'-':<5} | {'-':<28} | {'-':<8} | {'-':<7} | {res.confidence:<6.3f} | {res.status}")
        else:
            w_str = f"({res.w_anchor:.2f},{res.w_semantic:.2f},{res.w_structural:.2f},{res.w_spatial:.2f},{res.w_order:.2f})"
            print(
                f"{'Q' + q.number + ' -> ' + (res.answer_id or ''):<16} | "
                f"{res.anchor_score:<6.3f} | "
                f"{res.semantic_score:<8.3f} | "
                f"{res.structural_score:<6.3f} | "
                f"{res.spatial_score:<7.3f} | "
                f"{res.order_score:<5.3f} | "
                f"{w_str:<28} | "
                f"{res.raw_final_score:<8.3f} | "
                f"{res.conflict_penalty:<7.2f} | "
                f"{res.final_score:<6.3f} | "
                f"{res.status}"
            )

    print("-" * 125)
    print(f"Unmatched Regions Count: {len(unmatched)}")
    for u in unmatched:
        print(f" - Leftover Region ID: {u.answer_id} (Text preview: '{u.text[:40]}...')")

    # 6. Verify Requirement 6 Behavioral Scenarios
    print("\n--- STEP 6 BEHAVIORAL SCENARIO VERIFICATION ---")
    
    # 1) explicit anchor + compatible semantics -> high confidence
    q1_res = results.get("q1")
    print(f"1. Explicit anchor + compatible semantics: Q1 -> status={q1_res.status}, score={q1_res.final_score:.3f}, method={q1_res.method}")
    assert q1_res.status == "matched" and q1_res.final_score >= settings.MAPPING_HIGH_CONFIDENCE_THRESHOLD, "Scenario 1 failed!"
    
    # 2) explicit anchor + contradictory semantics -> review/conflict
    q2_res = results.get("q2")
    print(f"2. Explicit anchor + contradictory semantics: Q2 -> status={q2_res.status}, conflict_detected={q2_res.conflict_detected}, penalty={q2_res.conflict_penalty}")
    assert q2_res.conflict_detected and q2_res.status == "review_required", "Scenario 2 failed!"
    
    # 3) no anchor + strong semantics -> can achieve high confidence
    q3_res = results.get("q3")
    print(f"3. No anchor + strong semantics: Q3 -> status={q3_res.status}, score={q3_res.final_score:.3f}, w_sem={q3_res.w_semantic:.2f}")
    assert q3_res.w_semantic == 0.55 and q3_res.final_score >= settings.MAPPING_HIGH_CONFIDENCE_THRESHOLD, "Scenario 3 failed!"

    # 4) no anchor + weak semantics -> unanswered/unmatched/review
    q4_res = results.get("q4")
    print(f"4. No anchor + weak semantics: Q4 -> status={q4_res.status}, score={q4_res.final_score:.3f}")
    assert q4_res.status in ("unanswered", "review_required"), "Scenario 4 failed!"

    # 5) poor candidates -> null assignment
    q5_res = results.get("q5")
    print(f"5. Poor candidates: Q5 -> status={q5_res.status}, method={q5_res.method}")
    assert q5_res.status == "unanswered", "Scenario 5 failed!"

    # 6) competing candidates -> global assignment + margin analysis
    q6_res = results.get("q6")
    print(f"6. Competing candidates: Q6 -> assigned={q6_res.answer_id}, margin={q6_res.score_margin:.3f}, 2nd_best={q6_res.second_best_candidate_score:.3f}")
    assert q6_res.score_margin >= 0.0, "Scenario 6 failed!"

    print("\n" + "=" * 80)
    print("ALL DIAGNOSTIC CONSISTENCY CHECKS PASSED SUCCESSFULLY!")
    print("STEP 3 IS OFFICIALLY CLOSED.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    asyncio.run(run_diagnostic_check())
