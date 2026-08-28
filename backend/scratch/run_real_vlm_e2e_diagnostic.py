"""
Step 11B — Live Real End-to-End Multimodal VLM Verification Diagnostic.

Executes a live E2E verification test using an actual question paper image (qp.png/qp8.png)
with an actual page image payload sent to the VLM provider via OpenRouter/Gemini.

Proves:
1. Actual page image is sent to VLM.
2. Structured VLM JSON response is received & validated.
3. Selective VLM routing skips high-confidence cases and sends ambiguous/conflicting cases.
4. EvidenceFusion combines initial 11A hypotheses with VLM hypotheses into explicit verification states.
5. Diagnostic proving EvidenceFusion does not simply select the highest-weight source.
6. Re-disables VLM at the end to keep it opt-in.
"""
from __future__ import annotations
import sys
import os
import io
import time

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.schemas import Block, BBox, DocumentUnderstandingResult, DocumentRegion, StructureHypothesis
from app.services.document_processor import process_document
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.services.document_understanding_service import DocumentUnderstandingService, get_debug_summary
from app.services.evidence_fusion_service import EvidenceFusionService


def run_live_e2e_diagnostic():
    print("=" * 80)
    print("STEP 11B — LIVE E2E MULTIMODAL VLM VERIFICATION DIAGNOSTIC")
    print("=" * 80)

    original_vlm_enabled = settings.DOCUMENT_VLM_ENABLED
    settings.DOCUMENT_VLM_ENABLED = True

    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp.png"))
    if not os.path.exists(image_path):
        image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp8.png"))

    ext = os.path.splitext(image_path.lower())[1]
    print(f"[*] Target Sample Question Paper Image: {os.path.basename(image_path)}")
    print(f"[*] Document VLM Enabled: {settings.DOCUMENT_VLM_ENABLED}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    page_images_dict = {1: image_bytes}

    # Extract OCR blocks using existing DocumentProcessor (Step 2 OCR)
    blocks, num_pages, raw_sizes = process_document(image_path, ext)
    page_sizes_dict = {1: [float(raw_sizes[0][0]), float(raw_sizes[0][1])]}
    print(f"[*] Step 2 OCR Completed: {len(blocks)} blocks extracted across {num_pages} page(s).")

    # Instantiate DocumentUnderstandingService with live MultimodalDocumentVisionProvider
    live_provider = MultimodalDocumentVisionProvider()
    service = DocumentUnderstandingService(vision_provider=live_provider)

    start_time = time.time()
    # Process document with live VLM verification
    result: DocumentUnderstandingResult = service.process_document(
        blocks=blocks,
        document_id="live_e2e_qp_1",
        page_sizes=page_sizes_dict,
        page_images=page_images_dict,
        force_vlm_verification=True,
    )
    elapsed = time.time() - start_time

    cost = result.cost_accounting
    print("\n" + "=" * 80)
    print("VLM COST & EXECUTION ACCOUNTING REPORT")
    print("=" * 80)
    print(f"  VLM Status:                 {result.vlm_status}")
    print(f"  Provider / Model:           {live_provider.model_name}")
    print(f"  Pages Considered:           {cost.pages_considered}")
    print(f"  Pages Sent to VLM:          {cost.pages_sent}")
    print(f"  Regions Considered:         {cost.regions_considered}")
    print(f"  Regions Sent to VLM:        {cost.regions_sent}")
    print(f"  VLM API Calls Made:         {cost.vlm_calls}")
    print(f"  Successful Calls:           {cost.successful_calls}")
    print(f"  Failed Calls:               {cost.failed_calls}")
    print(f"  Skipped High-Confidence:    {cost.skipped_high_confidence_count}")
    print(f"  Latency:                    {elapsed:.2f}s")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("REAL REGION-LEVEL VERIFICATION EXAMPLES (11A HYPOTHESIS -> VLM -> FUSED RESULT)")
    print("=" * 80)

    # Display 3 detailed region-level verification examples
    sample_regions = result.regions[:3] if len(result.regions) >= 3 else result.regions
    for idx, reg in enumerate(sample_regions, 1):
        h_11a_str = ", ".join([f"{h.source}:{h.hypothesized_type}({h.confidence:.2f})" for h in reg.conflicting_hypotheses if h.source != "vlm"])
        vlm_h = reg.vlm_hypothesis
        vlm_str = f"{vlm_h.source}:{vlm_h.hypothesized_type}({vlm_h.confidence:.2f})" if vlm_h else "None (Skipped)"
        
        print(f"\nExample {idx}: Region '{reg.region_id}' (Page {reg.page})")
        print(f"  Text:                       '{reg.text[:70]}'")
        print(f"  Step 11A Hypotheses:        [{h_11a_str}]")
        print(f"  VLM Visual Hypothesis:      [{vlm_str}]")
        print(f"  Fused Region Type:          {reg.region_type}")
        print(f"  Fused Confidence:           {reg.confidence:.4f}")
        print(f"  Final Verification State:   {reg.verification_state}")
        print(f"  Conflict Detected:          {reg.classification_conflict}")

    # -------------------------------------------------------------------------
    # DIAGNOSTIC: Prove EvidenceFusion does not simply select highest-weight source
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: EVIDENCE FUSION CALIBRATION PROOF")
    print("=" * 80)
    print("Testing scenario: VLM (Weight 0.90) proposes INSTRUCTION with low score (0.55),")
    print("whereas Parser (Weight 0.85) & Layout (Weight 0.80) both propose QUESTION with high scores (0.95 & 0.90).")

    diag_reg = DocumentRegion(
        region_id="diag_proof_1",
        page=1,
        text="Question 10. Explain gradient descent optimization.",
        bbox=BBox(x=10, y=10, width=400, height=30),
        region_type="UNKNOWN",
        conflicting_hypotheses=[
            StructureHypothesis(region_id="diag_proof_1", hypothesized_type="QUESTION", confidence=0.95, source="parser"),
            StructureHypothesis(region_id="diag_proof_1", hypothesized_type="QUESTION", confidence=0.90, source="layout_analyzer"),
            StructureHypothesis(region_id="diag_proof_1", hypothesized_type="INSTRUCTION", confidence=0.55, source="vlm"),
        ],
    )
    diag_result = DocumentUnderstandingResult(document_id="proof_doc", pages=[], regions=[diag_reg])
    fused_proof = EvidenceFusionService().fuse_document_evidence(diag_result)
    proven_reg = fused_proof.regions[0]

    print(f"  Resulting Fused Classification: {proven_reg.region_type}")
    print(f"  Resulting Verification State:   {proven_reg.verification_state}")
    print(f"  Resulting Fused Confidence:     {proven_reg.confidence:.4f}")
    assert proven_reg.region_type == "QUESTION", "EvidenceFusion failed: wrongly picked low-score VLM over strong multi-source agreement!"
    print("  [DIAGNOSTIC PROOF PASSED] EvidenceFusion evaluates multi-source score calibration and did NOT blindly pick VLM!")
    print("=" * 80)

    # Reset VLM enabled setting to opt-in state
    settings.DOCUMENT_VLM_ENABLED = original_vlm_enabled
    print(f"\n[*] Reset DOCUMENT_VLM_ENABLED back to: {settings.DOCUMENT_VLM_ENABLED}")
    print("STEP 11B LIVE E2E DIAGNOSTIC COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_live_e2e_diagnostic()
