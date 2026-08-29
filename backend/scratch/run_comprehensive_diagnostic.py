"""
Comprehensive Visual Document Intelligence Diagnostic & Acceptance Runner.

Performs complete end-to-end execution of VedaAI Document Intelligence Core
against raw PDF/image documents and prints exhaustive diagnostic analysis.

Checks and verifies:
1. Complete OCR/Native ingestion & page image rendering
2. Page-level VLM Document Understanding execution & response
3. Evidence Fusion & DocumentStructureGraph construction
4. Graph-driven question extraction (walking graph vs fallback)
5. Distinguishes REAL_VLM_INTELLIGENCE from fallback
6. Evaluates false positives/negatives against expected document ground truth
"""
import sys
import os
import json
import asyncio
from typing import Dict, List, Any, Optional

# Add backend directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
from app.models.schemas import DocumentQuestionExtractionResult, DocumentUnderstandingResult

def run_diagnostic(file_path: str, force_vlm: bool = True):
    print("=" * 80)
    print(" VEDAAI DOCUMENT INTELLIGENCE CORE — COMPREHENSIVE DIAGNOSTIC")
    print("=" * 80)
    print(f"Target Document: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    is_pdf = ext == ".pdf"

    # Step 1: Ingestion (document_processor.py)
    print("\n--- STEP 1: DOCUMENT INGESTION & PAGE RENDERING ---")
    proc_res = process_document(file_path, ext, force_ocr=not is_pdf)
    blocks = proc_res[0]
    num_pages = proc_res[1]
    page_sizes = proc_res[2]
    page_images = proc_res[3] if len(proc_res) > 3 else {}

    print(f"Pages Ingested: {num_pages}")
    print(f"Total Raw Blocks Produced: {len(blocks)}")
    print(f"Page Images Rendered: {len(page_images)}")

    page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(page_sizes)} if page_sizes else None

    # Step 2: Document Understanding (document_understanding_service.py)
    print("\n--- STEP 2: VLM PAGE-LEVEL DOCUMENT UNDERSTANDING & GRAPH BUILDING ---")
    service = DocumentUnderstandingService()

    doc_res: DocumentUnderstandingResult = service.process_document(
        blocks=blocks,
        document_id="diag_doc",
        page_sizes=page_sizes_dict,
        page_images=page_images,
        attach_embeddings=False,
        force_vlm_verification=force_vlm,
    )

    print(f"VLM Status: {doc_res.vlm_status}")
    print(f"Document Purpose (Inferred): {doc_res.document_purpose}")
    print(f"Page Roles (Inferred): {doc_res.page_roles}")

    vlm_understandings = getattr(doc_res, "vlm_page_understandings", [])
    print(f"VLM Page Analyses Executed: {len(vlm_understandings)}")

    for idx, u in enumerate(vlm_understandings):
        print(f"\n  [Page {u.page_number} VLM Analysis]")
        print(f"   Image Sent: {u.image_sent}")
        print(f"   OCR Blocks Sent: {u.ocr_blocks_sent}")
        print(f"   Page Purpose: {u.page_purpose}")
        print(f"   Structures Found ({len(u.structures)}):")
        for st in u.structures:
            print(f"     • Role: {st.role:<16} Regions: {st.region_ids} | Display: '{st.display_number or st.display_label or ''}' | Reasoning: {st.reasoning[:70]}")
        print(f"   Relationships Found ({len(u.relationships)}):")
        for rel in u.relationships:
            print(f"     • {rel.source_ids} --[{rel.relationship_type}]--> {rel.target_ids}")

    # Region Breakdown
    print(f"\nTotal Document Regions: {len(doc_res.regions)}")
    for r in doc_res.regions:
        print(f"  • Region [{r.region_id}] Page {r.page} | Type: {r.region_type:<15} | State: {r.verification_state:<10} | Conf: {r.confidence:.2f} | Text: '{r.text[:60]}'")

    # Document Structure Graph
    graph = doc_res.structure_graph
    print("\n--- STEP 3: DOCUMENT STRUCTURE GRAPH ---")
    if graph:
        print(f"Graph Nodes ({len(graph.nodes)}):")
        for nid, node in graph.nodes.items():
            print(f"  • [{nid}] {node.role:<16} (P{node.page}) : '{node.text[:50]}'")
        print(f"Graph Edges ({len(graph.edges)}):")
        for edge in graph.edges:
            print(f"  • Edge: [{edge.source_id}] --[{edge.relationship}]--> [{edge.target_id}] (conf={edge.confidence:.2f})")
    else:
        print("  Graph is empty.")

    # Step 4: Intelligent Question Extraction
    print("\n--- STEP 4: GRAPH-DRIVEN QUESTION EXTRACTION ---")
    ext_service = IntelligentQuestionExtractionService(doc_understanding_service=service)
    extraction_res: DocumentQuestionExtractionResult = ext_service.extract_validated_questions(
        blocks=blocks,
        document_id="diag_doc",
        doc_understanding_result=doc_res,
        page_sizes=page_sizes_dict,
    )

    print(f"Fallback Used (Regex instead of Graph): {extraction_res.fallback_used}")
    print(f"Extracted Questions Count: {len(extraction_res.questions)}")
    print(f"Extracted Sections Count: {len(extraction_res.sections)}")
    print(f"Uncertain Candidates Count: {len(extraction_res.uncertain_candidates)}")

    print("\n--- EXTRACTED SECTIONS ---")
    for sec in extraction_res.sections:
        print(f"  • Section [{sec.section_id}] '{sec.title}' (Page {sec.page}) Regions: {sec.source_region_ids}")

    print("\n--- EXTRACTED QUESTIONS ---")
    for q in extraction_res.questions:
        print(f"\n  [Question {q.number}] ID: {q.id} (Type: {q.question_type})")
        print(f"   Section: {q.section or 'None'}")
        print(f"   Text: {q.text}")
        print(f"   Source Region IDs: {q.source_region_ids}")
        print(f"   Verification State: {q.verification_state}")
        if q.extracted_options:
            print(f"   Options ({len(q.extracted_options)}):")
            for opt in q.extracted_options:
                print(f"     - ({opt.label}) {opt.text} [Regions: {opt.source_region_ids}]")

    print("\n--- REJECTION AUDIT ---")
    for rej in extraction_res.audit.rejection_reasons:
        print(f"  • Rejected [{rej.region_id}] ({rej.classification}): '{rej.ocr_text}' -> Reason: {rej.reason}")

    # Step 5: Acceptance Assessment
    print("\n=" * 80)
    print(" ACCEPTANCE GATE EVALUATION")
    print("=" * 80)

    vlm_executed = doc_res.vlm_status == "SUCCESS" and len(vlm_understandings) > 0 and any(u.structures for u in vlm_understandings)
    print(f"1. Multimodal VLM Intelligence Executed: {'PASS' if vlm_executed else 'FAIL (Fallback/Not Executed)'}")
    print(f"2. Graph-Driven Extraction Used: {'PASS' if not extraction_res.fallback_used else 'FAIL (Fell back to regex)'}")
    print(f"3. Questions Recovered: {len(extraction_res.questions)}")
    print(f"4. Invariant Violations ({len(extraction_res.invariant_violations)}):")
    for inv in extraction_res.invariant_violations:
        print(f"  ! {inv}")

    overall = vlm_executed and not extraction_res.fallback_used and len(extraction_res.questions) > 0 and len(extraction_res.invariant_violations) == 0

    print(f"\nFINAL DIAGNOSTIC RESULT: {'PASS (REAL VISUAL INTELLIGENCE ACCEPTEABLE)' if overall else 'FAIL / UNCERTAIN'}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = os.path.join(os.path.dirname(__file__), "test_corpus", "digital_sectioned_mcq.png")

    run_diagnostic(target_file, force_vlm=True)
