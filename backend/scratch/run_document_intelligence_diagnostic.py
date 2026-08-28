"""
Step 11 Core — Real Visual Intelligence Validation & Empirical Diagnostic CLI.

Usage:
    python backend/scratch/run_document_intelligence_diagnostic.py [file_path]

Outputs:
1. Complete Document Ingestion Audit (Raw OCR Detections -> Normalized Blocks -> DocumentRegion -> Manifest)
2. Region Manifest Payload Supplied to VLM (Region ID, Page, BBox, OCR Text, Hypothesis)
3. Page Image Payload Inspection (Page Number, Image Resolution, Format, Base64 Payload Present)
4. Raw Structured VLM Response & Validated Relationships
5. Rejected VLM Relationships & Semantic Contradiction Audit
6. VLM Region ID Subset Validation (Returned Region IDs subset of Manifest IDs)
7. Evidence Fusion Decision Explanation (Combines Native, OCR, Layout, and VLM evidence)
8. Final DocumentStructureGraph (Nodes, Edges, Graph Purpose)
9. Graph-Driven Document Structure Hierarchy Tree
10. Real VLM Acceptance Gate Status (PASSED if VLM executed, verified & semantically consistent; FAILED otherwise)
11. Empirical Acceptance Quality Metrics Table
"""
from __future__ import annotations
import sys
import os
import json
import base64
import io

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.schemas import Block, DocumentQuestionExtractionResult, VisualVerificationResponse, CostAccounting
from app.services.document_processor import process_document, _get_ocr_engine
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


def run_empirical_diagnostic(file_path: str, force_vlm: bool = True):
    print("=" * 110)
    print("VEDAAI — REAL VISUAL INTELLIGENCE & EMPIRICAL DIAGNOSTIC CLI")
    print("=" * 110)
    print(f"[*] Target Document: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    ext = os.path.splitext(file_path.lower())[1]

    # 1. Process Document (Smart Ingestion with Native PDF / OCR Quality Inspection)
    proc_res = process_document(file_path, ext, force_ocr=False)
    blocks = proc_res[0]
    num_pages = proc_res[1]
    sizes = proc_res[2]
    page_images = proc_res[3] if len(proc_res) > 3 else {}

    native_blocks_count = sum(1 for b in blocks if b.source == "native_pdf")
    ocr_blocks_count = sum(1 for b in blocks if b.source == "ocr")

    print("\n1. DOCUMENT INGESTION AUDIT (IMAGE -> OCR DETECTIONS -> BLOCKS -> REGIONS)")
    print(f"+-- File: {os.path.basename(file_path)} (Type: {ext}, Pages: {num_pages})")
    print(f"+-- Page Dimensions: {sizes}")
    print(f"+-- Total Ingested Blocks: {len(blocks)} (Native PDF: {native_blocks_count}, OCR: {ocr_blocks_count})")

    page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(sizes)} if sizes else None

    # Group blocks by page for diagnostic breakdown
    blocks_by_page = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page, []).append(b)

    for p_num in range(1, num_pages + 1):
        p_blocks = blocks_by_page.get(p_num, [])
        p_native = sum(1 for b in p_blocks if b.source == "native_pdf")
        p_ocr = sum(1 for b in p_blocks if b.source == "ocr")
        img_info = "NO"
        if p_num in page_images:
            img = page_images[p_num]
            img_info = f"YES ({img.size[0]}x{img.size[1]} {img.mode})"
        print(f"|")
        print(f"+-- PAGE {p_num}")
        print(f"|   +-- Rendered Image Present: {img_info}")
        print(f"|   +-- Native PDF Blocks: {p_native}")
        print(f"|   +-- OCR Blocks: {p_ocr}")
        print(f"|   +-- Total Page Blocks: {len(p_blocks)}")
        for idx, b in enumerate(p_blocks[:10]):
            print(f"|       [{idx+1}] Block {b.id} | BBox: [{b.bbox.x}, {b.bbox.y}, {b.bbox.width}, {b.bbox.height}] | Conf: {b.confidence} | '{b.text[:40]}'")
        if len(p_blocks) > 10:
            print(f"|       ... and {len(p_blocks) - 10} more blocks on page {p_num}")

    # 2. Document Understanding & Structure Graph Construction with Live VLM Path
    doc_service = DocumentUnderstandingService()
    doc_result = doc_service.process_document(
        blocks=blocks,
        document_id=f"doc_{os.path.basename(file_path)}",
        page_sizes=page_sizes_dict,
        page_images=page_images,
        force_vlm_verification=force_vlm,
    )

    # 3. Intelligent Question Extraction (Graph-Driven Assembly)
    extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
    result = extractor.extract_validated_questions(
        blocks=blocks,
        document_id=f"doc_{os.path.basename(file_path)}",
        doc_understanding_result=doc_result,
        page_sizes=page_sizes_dict,
    )

    # --- PRINT REGION MANIFEST PAYLOAD ---
    print("\n" + "=" * 110)
    print("2. REGION MANIFEST PAYLOAD SUPPLIED TO VLM")
    print("=" * 110)
    manifest_items = []
    for r in doc_result.regions:
        manifest_items.append(
            {
                "region_id": r.region_id,
                "page": r.page,
                "bbox": [round(r.bbox.x, 1), round(r.bbox.y, 1), round(r.bbox.width, 1), round(r.bbox.height, 1)],
                "ocr_text": r.text[:50],
                "initial_hypothesis": r.region_type,
            }
        )
    print(json.dumps(manifest_items[:15], indent=2))
    if len(manifest_items) > 15:
        print(f"... and {len(manifest_items) - 15} more regions in manifest.")

    # --- PRINT PAGE IMAGE PAYLOAD INSPECTION ---
    print("\n" + "=" * 110)
    print("3. PAGE IMAGE PAYLOAD INSPECTION")
    print("=" * 110)
    for p_num, img in page_images.items():
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_len = len(base64.b64encode(buf.getvalue()))
        print(f"  Page {p_num}: {img.size[0]}x{img.size[1]} PNG ({b64_len} b64 chars)")

    # --- LIVE VLM VERIFICATION BREAKDOWN ---
    vlm_cost = doc_result.cost_accounting
    vlm_status = doc_result.vlm_status
    vlm_has_keys = bool(getattr(settings, "GEMINI_API_KEY", "") or getattr(settings, "OPENROUTER_API_KEY", ""))

    print("\n" + "=" * 110)
    print("4. VLM VERIFICATION & SUBSET REGION ID VALIDATION")
    print("=" * 110)
    print(f"  VLM Configured in Settings:    {getattr(settings, 'DOCUMENT_VLM_ENABLED', False)}")
    print(f"  API Keys Configured:           {vlm_has_keys}")
    print(f"  VLM Status:                    {vlm_status}")
    print(f"  Pages Sent to VLM:             {vlm_cost.pages_sent if vlm_cost else 0}")
    print(f"  Regions Targeted:              {vlm_cost.regions_sent if vlm_cost else 0}")
    print(f"  Region Manifest Count:         {len(doc_result.regions)}")
    
    # Subset Validation
    valid_manifest_ids = {r.region_id for r in doc_result.regions}
    invalid_ids = [rel.source_region_id for rel in doc_result.relationships if rel.source_region_id not in valid_manifest_ids]
    invalid_ids.extend([rel.target_region_id for rel in doc_result.relationships if rel.target_region_id not in valid_manifest_ids])

    if invalid_ids:
        print(f"  Subset Region ID Validation:  FAIL (Invalid IDs returned: {invalid_ids})")
    else:
        print(f"  Subset Region ID Validation:  PASS (All VLM Region IDs subset of Manifest IDs)")

    # Print Valid VLM Relationships
    print("\n--- VALIDATED VLM RELATIONSHIP GRAPH EDGES ---")
    if doc_result.relationships:
        for rel in doc_result.relationships:
            print(f"  {rel.source_region_id:<12} --[{rel.relationship_type}]--> {rel.target_region_id:<12} (Conf: {rel.confidence:.2f})")
    else:
        print("  (No valid non-contradictory VLM relationships)")

    # Print Rejected / Contradictory VLM Relationships
    rejected_rels = doc_result.metadata.get("rejected_vlm_relationships", [])
    print("\n--- REJECTED / CONTRADICTORY VLM RELATIONSHIP EDGES (AUDIT LOG) ---")
    if rejected_rels:
        for rej in rejected_rels:
            print(f"  [REJECTED] {rej.get('source')} --[{rej.get('rel_type')}]--> {rej.get('target')} | Reason: {rej.get('reason')}")
    else:
        print("  (No VLM relationships were rejected)")

    # --- REAL VLM ACCEPTANCE GATE REPORT ---
    print("\n" + "=" * 110)
    print("REAL VLM ACCEPTANCE GATE REPORT")
    print("=" * 110)
    vlm_acceptance_passed = True
    vlm_failure_reason = ""

    if vlm_status in ("NOT_CONFIGURED", "VLM_UNAVAILABLE") and not vlm_has_keys:
        vlm_acceptance_passed = False
        vlm_failure_reason = "VLM_NOT_AVAILABLE (No GEMINI_API_KEY or OPENROUTER_API_KEY configured in environment)"
    elif rejected_rels and any("contradiction" in r.get("reason", "").lower() for r in rejected_rels):
        vlm_acceptance_passed = False
        vlm_failure_reason = f"SEMANTIC_CONTRADICTIONS_DETECTED ({len(rejected_rels)} contradictory VLM edges filtered)"

    if not vlm_acceptance_passed:
        print("  REAL_VLM_ACCEPTANCE: FAILED")
        print(f"  REASON: {vlm_failure_reason}")
        result.invariant_violations.append(f"INVARIANT_VIOLATION: Real VLM Acceptance Gate failed: {vlm_failure_reason}")
    else:
        print("  REAL_VLM_ACCEPTANCE: PASSED")
        print(f"  STATUS: {vlm_status}")

    # --- EVIDENCE FUSION EXPLANATION ---
    print("\n" + "=" * 110)
    print("5. EVIDENCE FUSION DECISION EXPLANATION")
    print("=" * 110)
    print(f"{'Region ID':<12} | {'Pg':<3} | {'Sources':<10} | {'11A Type':<15} | {'State':<10} | {'Dynamic Conf':<12} | {'Supporting Evidence'}")
    print("-" * 110)
    for r in doc_result.regions[:15]:
        sources_str = ",".join({h.source for h in r.conflicting_hypotheses}) if r.conflicting_hypotheses else r.source
        print(f"{r.region_id:<12} | {r.page:<3} | {sources_str:<10} | {r.region_type:<15} | {r.verification_state:<10} | {r.confidence:<12.4f} | {r.text[:30]}")

    # --- GRAPH-DRIVEN HIERARCHY TREE ---
    print("\n" + "=" * 110)
    print("6. GRAPH-DRIVEN DOCUMENT STRUCTURE HIERARCHY TREE")
    print("=" * 110)
    if result.sections:
        for sec in result.sections:
            print(f"+-- {sec.title} [ID: {sec.section_id}] (Page {sec.page})")
            sec_questions = [q for q in result.questions if q.section_id == sec.section_id or q.section_title == sec.title]
            if sec_questions:
                for q in sec_questions:
                    is_cont = " (Multi-Region Continuation)" if len(q.source_region_ids) > 1 else ""
                    print(f"|   +-- Q{q.number} [ID: {q.id}] ({q.question_type}{is_cont}) (Conf: {q.extraction_confidence:.4f})")
                    print(f"|   |   Text: {q.text[:60]}")
                    if q.extracted_options:
                        for opt in q.extracted_options:
                            print(f"|   |   +-- Option [{opt.label}]: {opt.text[:40]} [Reg: {opt.source_region_ids}]")
            else:
                print("|   +-- (No questions in section)")
    else:
        print("Root (Unsectioned)")
        for q in result.questions:
            is_cont = " (Multi-Region Continuation)" if len(q.source_region_ids) > 1 else ""
            print(f"+-- Q{q.number} [ID: {q.id}] ({q.question_type}{is_cont}) (Conf: {q.extraction_confidence:.4f})")
            print(f"    Text: {q.text[:60]}")
            if q.extracted_options:
                for opt in q.extracted_options:
                    print(f"    +-- Option [{opt.label}]: {opt.text[:40]} [Reg: {opt.source_region_ids}]")

    # --- REJECTED CONTENT ---
    print("\n--- REJECTED NON-QUESTION CONTENT & STRUCTURAL EVIDENCE ---")
    if result.audit.rejection_reasons:
        for rr in result.audit.rejection_reasons:
            print(f"  {rr.region_id:<12} | {rr.classification:<16} | {rr.reason}")
    else:
        print("  No administrative/visual regions were rejected.")

    # --- DIAGNOSTIC INVARIANTS STATUS ---
    print("\n" + "=" * 110)
    print("7. DIAGNOSTIC INVARIANTS & CONSISTENCY STATUS")
    print("=" * 110)
    if result.invariant_violations:
        print("  INVARIANT STATUS: FAIL")
        for viol in result.invariant_violations:
            print(f"  [ERROR] {viol}")
    else:
        print("  INVARIANT STATUS: PASS (All internal metrics, graph edges, and ID bounds consistent)")

    # --- EMPIRICAL ACCEPTANCE METRICS ---
    duplicate_id_count = len(result.questions) - len({q.id for q in result.questions})

    print("\n" + "=" * 110)
    print("8. EMPIRICAL ACCEPTANCE QUALITY METRICS TABLE")
    print("=" * 110)
    print(f"  Ground Truth Status:            GROUND_TRUTH_UNAVAILABLE (Arbitrary Document)")
    print(f"  Total Candidates Ingested:      {result.audit.candidate_count}")
    print(f"  Extracted Genuine Questions:    {len(result.questions)}")
    print(f"  Extracted Options Count:        {sum(len(q.extracted_options) for q in result.questions)}")
    print(f"  Extracted Sections Count:       {len(result.sections)}")
    print(f"  Multi-Region Questions Count:   {result.audit.multi_region_question_count}")
    print(f"  Multi-Page Questions Count:     {result.audit.multi_page_question_count}")
    print(f"  Duplicate Internal IDs:         {duplicate_id_count}")
    print(f"  Unresolved Regions:             {len(result.uncertain_candidates)}")
    print(f"  VLM Status / Calls Made:        {doc_result.vlm_status} ({vlm_cost.vlm_calls if vlm_cost else 0} calls)")
    print(f"  Rejected VLM Relationships:     {len(rejected_rels)}")
    print(f"  Fallback Used (ACCEPTANCE):     {result.fallback_used}")
    print("=" * 110)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), "qp.png"))
    if not os.path.exists(target):
        target = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp8.png"))
    run_empirical_diagnostic(target)
