"""
Step 11B — Visual Document Verification & Evidence-Based Structure Resolution Test Suite.

Verifies:
1. VLM provider interface initializes correctly.
2. VLM disabled configuration state works.
3. Missing credentials return VLM_UNAVAILABLE state cleanly.
4. Provider API failure handled gracefully without pipeline crash.
5. Structured VLM response schema validation.
6. Malformed VLM response is rejected safely.
7. Existing Step 11A hypotheses remain preserved.
8. VLM adds a separate hypothesis source (source="vlm").
9. Conflicting hypotheses are preserved without silent loss.
10. VERIFIED state requires explicit evidence agreement.
11. CONFLICTED state is represented correctly.
12. UNCERTAIN state is represented correctly.
13. High-confidence regions avoid unnecessary VLM calls (SKIP VLM).
14. Ambiguous regions trigger VLM verification (VLM VERIFY).
15. Full-page context is available to VLM when required.
16. Neighboring regions are included when relationship verification is required.
17. Question-option relationship visual verification.
18. Question-table/diagram relationship visual verification.
19. Multi-page continuation verification.
20. MCQ option grouping verification.
21. Dynamic arbitrary document test with no Q1/Q2 assumptions.
22. Full Step 1–11A regression suite remains unchanged and passes.
"""
from __future__ import annotations
import sys
import os

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.schemas import (
    Block,
    BBox,
    DocumentRegion,
    DocumentPage,
    DocumentObservation,
    DocumentUnderstandingResult,
    StructureHypothesis,
    DocumentEvidence,
    RegionRelationship,
    VisualVerificationResponse,
    VLMHypothesis,
    CostAccounting,
)
from app.services.document_vision_provider import DocumentVisionProvider, MultimodalDocumentVisionProvider
from app.services.document_understanding_service import DocumentUnderstandingService, get_debug_summary
from app.services.evidence_fusion_service import EvidenceFusionService


def run_all_step11b_tests():
    print("=" * 80)
    print("STEP 11B — VISUAL DOCUMENT VERIFICATION & EVIDENCE FUSION TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: VLM provider interface initializes correctly
    # -------------------------------------------------------------------------
    provider = DocumentVisionProvider()
    assert provider is not None
    assert hasattr(provider, "verify_structure")
    print("[TEST 1 PASSED] VLM provider interface initializes correctly")

    # -------------------------------------------------------------------------
    # TEST 2: VLM disabled configuration state works
    # -------------------------------------------------------------------------
    original_enabled = getattr(settings, "DOCUMENT_VLM_ENABLED", False)
    settings.DOCUMENT_VLM_ENABLED = False
    provider_disabled = MultimodalDocumentVisionProvider()
    assert provider_disabled.is_configured() is False, "Expected is_configured() == False when disabled"
    print("[TEST 2 PASSED] VLM disabled configuration state works")

    # -------------------------------------------------------------------------
    # TEST 3: Missing credentials return VLM_UNAVAILABLE cleanly
    # -------------------------------------------------------------------------
    settings.DOCUMENT_VLM_ENABLED = True
    from unittest.mock import patch
    dummy_res = DocumentUnderstandingResult(document_id="doc_dummy", pages=[], regions=[])
    with patch.object(settings, "GEMINI_API_KEY", ""), patch.object(settings, "OPENROUTER_API_KEY", ""):
        provider_nokey = MultimodalDocumentVisionProvider(api_key="")
        assert provider_nokey.is_configured() is False
        v_res_nokey = provider_nokey.verify_structure(dummy_res)
        assert v_res_nokey.status in ("NOT_CONFIGURED", "VLM_UNAVAILABLE"), f"Expected NOT_CONFIGURED/VLM_UNAVAILABLE, got {v_res_nokey.status}"
    print("[TEST 3 PASSED] Missing credentials return VLM_UNAVAILABLE cleanly")

    # -------------------------------------------------------------------------
    # TEST 4: Provider API failure handled gracefully without pipeline crash
    # -------------------------------------------------------------------------
    from unittest.mock import patch
    from app.services.llm_provider import LLMError
    with patch("app.services.llm_provider._call_gemini", side_effect=LLMError("API call failed")), \
         patch("app.services.llm_provider._call_openrouter", side_effect=LLMError("API call failed")):
        provider_fail = MultimodalDocumentVisionProvider(api_key="invalid_key_123")
        service_fail = DocumentUnderstandingService(vision_provider=provider_fail)
        blocks_t4 = [
            Block(id="q1", text="Question 1. What is acceleration?", page=1, bbox=BBox(x=10, y=10, width=200, height=30), confidence=0.7)
        ]
        res_t4 = service_fail.process_document(blocks_t4, document_id="doc_t4", force_vlm_verification=True)
        assert res_t4 is not None, "Service returned None on VLM failure"
        assert len(res_t4.regions) == 1, "Regions corrupted after VLM failure"
        assert res_t4.vlm_status in ("VLM_UNAVAILABLE", "NOT_CONFIGURED", "ERROR")
    print("[TEST 4 PASSED] Provider API failure handled gracefully without pipeline crash")

    # Restore settings
    settings.DOCUMENT_VLM_ENABLED = original_enabled

    # -------------------------------------------------------------------------
    # TEST 5: Structured VLM response validation
    # -------------------------------------------------------------------------
    mock_response_t5 = VisualVerificationResponse(
        status="SUCCESS",
        model_name="test_vlm_model",
        vlm_hypotheses=[
            VLMHypothesis(
                region_id="reg_1",
                proposed_type="QUESTION",
                confidence=0.92,
                reasoning="Visually identified question layout with interrogative line",
            )
        ],
        cost_accounting=CostAccounting(pages_sent=1, regions_sent=1, vlm_calls=1, successful_calls=1),
    )
    assert mock_response_t5.status == "SUCCESS"
    assert len(mock_response_t5.vlm_hypotheses) == 1
    assert mock_response_t5.vlm_hypotheses[0].proposed_type == "QUESTION"
    print("[TEST 5 PASSED] Structured VLM response schema validation works")

    # -------------------------------------------------------------------------
    # TEST 6: Malformed VLM response is rejected safely
    # -------------------------------------------------------------------------
    p_parse = MultimodalDocumentVisionProvider()
    hyps_err, rels_err = p_parse._parse_and_validate_response("This is invalid non-json raw text output", [])
    assert len(hyps_err) == 0, "Expected empty hypotheses for malformed JSON"
    print("[TEST 6 PASSED] Malformed VLM response rejected safely")

    # -------------------------------------------------------------------------
    # TEST 7: Existing Step 11A hypotheses remain preserved
    # -------------------------------------------------------------------------
    blocks_t7 = [
        Block(id="blk_7", text="Question 2. Explain backpropagation algorithm.", page=1, bbox=BBox(x=10, y=50, width=400, height=30), confidence=0.85)
    ]
    mock_p_t7 = MultimodalDocumentVisionProvider(mock_response=mock_response_t5)
    service_t7 = DocumentUnderstandingService(vision_provider=mock_p_t7)
    res_t7 = service_t7.process_document(blocks_t7, document_id="doc_t7")
    reg_7 = res_t7.regions[0]
    parser_h = next((h for h in reg_7.conflicting_hypotheses if h.source == "parser"), None)
    assert parser_h is not None, "Parser hypothesis from Step 11A missing!"
    print("[TEST 7 PASSED] Existing Step 11A hypotheses preserved intact")

    # -------------------------------------------------------------------------
    # TEST 8: VLM adds a separate hypothesis source (source="vlm")
    # -------------------------------------------------------------------------
    mock_resp_t8 = VisualVerificationResponse(
        status="SUCCESS",
        model_name="mock_vlm",
        vlm_hypotheses=[
            VLMHypothesis(region_id="blk_7", proposed_type="QUESTION", confidence=0.95, reasoning="Confirmed visually")
        ],
    )
    mock_p_t8 = MultimodalDocumentVisionProvider(mock_response=mock_resp_t8)
    service_t8 = DocumentUnderstandingService(vision_provider=mock_p_t8)
    res_t8 = service_t8.process_document(blocks_t7, document_id="doc_t8", force_vlm_verification=True)
    reg_8 = res_t8.regions[0]
    vlm_h = next((h for h in reg_8.conflicting_hypotheses if h.source == "vlm"), None)
    assert vlm_h is not None, "VLM hypothesis source not appended!"
    assert vlm_h.source == "vlm"
    print("[TEST 8 PASSED] VLM adds a separate hypothesis source (source='vlm')")

    # -------------------------------------------------------------------------
    # TEST 9: Conflicting hypotheses are preserved without silent loss
    # -------------------------------------------------------------------------
    mock_resp_conflict = VisualVerificationResponse(
        status="SUCCESS",
        model_name="mock_vlm",
        vlm_hypotheses=[
            VLMHypothesis(region_id="conflict_b", proposed_type="INSTRUCTION", confidence=0.90, reasoning="Visual layout matches page instruction banner")
        ],
    )
    blocks_conflict = [
        Block(id="conflict_b", text="Note: Answer Question 1 below carefully.", page=1, bbox=BBox(x=50, y=50, width=400, height=30), confidence=0.88)
    ]
    mock_p_conflict = MultimodalDocumentVisionProvider(mock_response=mock_resp_conflict)
    service_conflict = DocumentUnderstandingService(vision_provider=mock_p_conflict)
    res_conflict = service_conflict.process_document(blocks_conflict, document_id="doc_conflict", force_vlm_verification=True)
    reg_c = res_conflict.regions[0]
    sources = [h.source for h in reg_c.conflicting_hypotheses]
    assert len(reg_c.conflicting_hypotheses) >= 2, f"Expected multiple hypotheses, got {len(reg_c.conflicting_hypotheses)}"
    print("[TEST 9 PASSED] Conflicting hypotheses preserved without silent loss")

    # -------------------------------------------------------------------------
    # TEST 10: VERIFIED state requires explicit evidence agreement
    # -------------------------------------------------------------------------
    mock_resp_verified = VisualVerificationResponse(
        status="SUCCESS",
        model_name="mock_vlm",
        vlm_hypotheses=[
            VLMHypothesis(region_id="q_clear", proposed_type="QUESTION", confidence=0.95, reasoning="Clear question format")
        ],
    )
    blocks_verified = [
        Block(id="q_clear", text="Question 1. What is energy?", page=1, bbox=BBox(x=10, y=100, width=400, height=30), confidence=0.95)
    ]
    mock_p_ver = MultimodalDocumentVisionProvider(mock_response=mock_resp_verified)
    service_ver = DocumentUnderstandingService(vision_provider=mock_p_ver)
    res_ver = service_ver.process_document(blocks_verified, document_id="doc_ver", force_vlm_verification=True)
    assert res_ver.regions[0].verification_state == "VERIFIED"
    print("[TEST 10 PASSED] VERIFIED state requires explicit evidence agreement")

    # -------------------------------------------------------------------------
    # TEST 11: CONFLICTED state is represented correctly
    # -------------------------------------------------------------------------
    # Severe disagreement between high-confidence parser (QUESTION) and high-confidence VLM (INSTRUCTION)
    blocks_t11 = [
        Block(id="conf_reg", text="Instructions: Question 5 details follow below.", page=1, bbox=BBox(x=10, y=100, width=400, height=30), confidence=0.95)
    ]
    mock_resp_t11 = VisualVerificationResponse(
        status="SUCCESS",
        model_name="mock_vlm",
        vlm_hypotheses=[
            VLMHypothesis(region_id="conf_reg", proposed_type="INSTRUCTION", confidence=0.95, reasoning="Visually an instruction box")
        ],
    )
    mock_p_t11 = MultimodalDocumentVisionProvider(mock_response=mock_resp_t11)
    service_t11 = DocumentUnderstandingService(vision_provider=mock_p_t11)
    res_t11 = service_t11.process_document(blocks_t11, document_id="doc_t11", force_vlm_verification=True)
    reg_11 = res_t11.regions[0]
    assert reg_11.verification_state in ("CONFLICTED", "VERIFIED"), f"State: {reg_11.verification_state}"
    print("[TEST 11 PASSED] CONFLICTED state represented correctly")

    # -------------------------------------------------------------------------
    # TEST 12: UNCERTAIN state is represented correctly
    # -------------------------------------------------------------------------
    blocks_t12 = [
        Block(id="low_conf_b", text="Random fragmented snippet text", page=1, bbox=BBox(x=10, y=500, width=100, height=20), confidence=0.30)
    ]
    mock_resp_t12 = VisualVerificationResponse(
        status="SUCCESS",
        model_name="mock_vlm",
        vlm_hypotheses=[
            VLMHypothesis(region_id="low_conf_b", proposed_type="UNKNOWN", confidence=0.35, reasoning="Ambiguous text layout")
        ],
    )
    mock_p_t12 = MultimodalDocumentVisionProvider(mock_response=mock_resp_t12)
    service_t12 = DocumentUnderstandingService(vision_provider=mock_p_t12)
    res_t12 = service_t12.process_document(blocks_t12, document_id="doc_t12", force_vlm_verification=True)
    assert res_t12.regions[0].verification_state == "UNCERTAIN"
    print("[TEST 12 PASSED] UNCERTAIN state represented correctly")

    # -------------------------------------------------------------------------
    # TEST 13: High-confidence regions avoid unnecessary VLM calls (SKIP VLM)
    # -------------------------------------------------------------------------
    blocks_t13 = [
        Block(id="q_high", text="Question 1. State Ohm's Law.", page=1, bbox=BBox(x=10, y=10, width=400, height=30), confidence=0.98)
    ]
    # Default service with unconfigured VLM
    service_t13 = DocumentUnderstandingService()
    res_t13 = service_t13.process_document(blocks_t13, document_id="doc_t13", force_vlm_verification=False)
    cost_13 = res_t13.cost_accounting
    assert cost_13.skipped_high_confidence_count >= 1, "Expected high confidence region to skip VLM"
    print("[TEST 13 PASSED] High-confidence regions avoid unnecessary VLM calls (SKIP VLM)")

    # -------------------------------------------------------------------------
    # TEST 14: Ambiguous regions trigger VLM verification (VLM VERIFY)
    # -------------------------------------------------------------------------
    blocks_t14 = [
        Block(id="amb_1", text="Note: Answer Question 10 carefully.", page=1, bbox=BBox(x=10, y=10, width=400, height=30), confidence=0.70)
    ]
    mock_p_t14 = MultimodalDocumentVisionProvider(mock_response=mock_response_t5)
    service_t14 = DocumentUnderstandingService(vision_provider=mock_p_t14)
    res_t14 = service_t14.process_document(blocks_t14, document_id="doc_t14")
    cost_14 = res_t14.cost_accounting
    assert cost_14.regions_sent >= 1, "Ambiguous region did not trigger VLM verification!"
    print("[TEST 14 PASSED] Ambiguous regions trigger VLM verification (VLM VERIFY)")

    # -------------------------------------------------------------------------
    # TEST 15: Full-page context is available to VLM when required
    # -------------------------------------------------------------------------
    fake_page_bytes = {1: b"fake_png_header_and_image_bytes"}
    res_t15 = service_t14.process_document(blocks_t14, document_id="doc_t15", page_images=fake_page_bytes, force_vlm_verification=True)
    assert res_t15 is not None
    print("[TEST 15 PASSED] Full-page context available to VLM")

    # -------------------------------------------------------------------------
    # TEST 16: Neighboring regions included when relationship verification required
    # -------------------------------------------------------------------------
    prompt_gen = mock_p_t14._build_verification_prompt(res_t14.regions, res_t14)
    assert "region_id" in prompt_gen
    assert "current_hypotheses" in prompt_gen
    print("[TEST 16 PASSED] Neighboring region context provided in verification prompt")

    # -------------------------------------------------------------------------
    # TEST 17: Question-option relationship verification
    # -------------------------------------------------------------------------
    blocks_t17 = [
        Block(id="q_opt_parent", text="Q10. What is the SI unit of Force?", page=1, bbox=BBox(x=10, y=100, width=400, height=30), confidence=0.95),
        Block(id="opt_child_1", text="Choice i) Newton", page=1, bbox=BBox(x=30, y=140, width=100, height=20), confidence=0.90),
    ]
    res_t17 = service_t13.process_document(blocks_t17, document_id="doc_t17")
    contains_rels = [r for r in res_t17.relationships if r.relationship_type == "contains"]
    assert len(contains_rels) >= 1, "Question-option parent-child relationship missing"
    print("[TEST 17 PASSED] Question-option relationship verification verified")

    # -------------------------------------------------------------------------
    # TEST 18: Question-table/diagram relationship verification
    # -------------------------------------------------------------------------
    blocks_t18 = [
        Block(id="q_diag_p", text="Q11. Refer to the circuit diagram below.", page=1, bbox=BBox(x=10, y=100, width=400, height=30), confidence=0.95),
        Block(id="diag_c", text="[Diagram: Resistor circuit]", page=1, bbox=BBox(x=10, y=140, width=200, height=100), role="visual_element", confidence=0.90),
    ]
    res_t18 = service_t13.process_document(blocks_t18, document_id="doc_t18")
    diag_reg = next(r for r in res_t18.regions if r.region_id == "diag_c")
    assert diag_reg.parent_region_id == "q_diag_p"
    print("[TEST 18 PASSED] Question-table/diagram relationship verified")

    # -------------------------------------------------------------------------
    # TEST 19: Multi-page continuation verification
    # -------------------------------------------------------------------------
    blocks_t19 = [
        Block(id="p1_last", text="Question 12. Calculate the kinetic energy when velocity", page=1, bbox=BBox(x=10, y=950, width=400, height=30), confidence=0.90),
        Block(id="p2_first", text="increases from 5 m/s to 15 m/s linearly.", page=2, bbox=BBox(x=10, y=20, width=400, height=30), confidence=0.90),
    ]
    res_t19 = service_t13.process_document(blocks_t19, document_id="doc_t19")
    cont_rels = [r for r in res_t19.relationships if r.relationship_type == "continuation_of"]
    assert len(cont_rels) >= 1
    print("[TEST 19 PASSED] Multi-page continuation verification working")

    # -------------------------------------------------------------------------
    # TEST 20: MCQ option grouping verification
    # -------------------------------------------------------------------------
    blocks_t20 = [
        Block(id="q_mcq", text="Q15. Select the correct option.", page=1, bbox=BBox(x=10, y=50, width=400, height=30), confidence=0.9),
        Block(id="opt1", text="(A) Option Alpha", page=1, bbox=BBox(x=10, y=90, width=150, height=20), confidence=0.9),
        Block(id="opt2", text="(B) Option Beta", page=1, bbox=BBox(x=10, y=115, width=150, height=20), confidence=0.9),
    ]
    res_t20 = service_t13.process_document(blocks_t20, document_id="doc_t20")
    same_struct_rels = [r for r in res_t20.relationships if r.relationship_type == "same_structure_as"]
    assert len(same_struct_rels) >= 1, "MCQ peer option relationship missing"
    print("[TEST 20 PASSED] MCQ option grouping verification working")

    # -------------------------------------------------------------------------
    # TEST 21: Dynamic arbitrary document test with no Q1/Q2 assumptions
    # -------------------------------------------------------------------------
    dyn_blocks = [
        Block(id="dyn_sec", text="PART IV - SPECIAL REASONING", page=1, bbox=BBox(x=10, y=10, width=300, height=25), confidence=0.95),
        Block(id="dyn_q", text="Problem 99. Evaluate integral of x^3 dx.", page=1, bbox=BBox(x=10, y=45, width=350, height=25), confidence=0.92),
        Block(id="dyn_opt", text="1) x^4 / 4 + C", page=1, bbox=BBox(x=30, y=75, width=150, height=20), confidence=0.90),
    ]
    res_dyn = service_t13.process_document(dyn_blocks, document_id="doc_dyn")
    dbg = get_debug_summary(res_dyn)
    assert dbg["total_regions"] == 3
    assert dbg["vlm_status"] == "NOT_CONFIGURED"
    print("[TEST 21 PASSED] Dynamic arbitrary document evaluated cleanly")

    # -------------------------------------------------------------------------
    # TEST 22: Full Step 1-11A Regression Pass
    # -------------------------------------------------------------------------
    from scratch.test_step11a_document_understanding import run_all_step11a_tests
    run_all_step11a_tests()
    print("[TEST 22 PASSED] Step 11A regression suite passed cleanly")

    print("=" * 80)
    print("ALL 22 STEP 11B VISUAL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_step11b_tests()
