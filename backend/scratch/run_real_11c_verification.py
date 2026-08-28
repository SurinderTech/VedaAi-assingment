"""
Step 11C — Real Document Intelligent Question Extraction Verification Diagnostic.

Executes Step 11C extraction on real sample question paper image (qp.png / qp8.png).
Prints:
1. Region-level diagnostic table (Region ID, OCR Text, 11A Type, 11B State, Final Type, Question ID, Confidence).
2. Question Extraction Summary table (Order, Number, Type, Text, Options, Pages, Confidence).
3. DocumentQuestionExtractionResult summary and ExtractionAudit breakdown.
"""
from __future__ import annotations
import sys
import os

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import Block, DocumentQuestionExtractionResult
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


def run_real_document_verification():
    print("=" * 110)
    print("STEP 11C — REAL QUESTION PAPER DOCUMENT EXTRACTION DIAGNOSTIC")
    print("=" * 110)

    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp.png"))
    if not os.path.exists(image_path):
        image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp8.png"))

    ext = os.path.splitext(image_path.lower())[1]
    print(f"[*] Target Sample Question Paper Image: {os.path.basename(image_path)}")

    # 1. Step 2 OCR Block Extraction
    blocks, num_pages, raw_sizes = process_document(image_path, ext)
    page_sizes_dict = {1: [float(raw_sizes[0][0]), float(raw_sizes[0][1])]}
    print(f"[*] Step 2 OCR Completed: {len(blocks)} blocks extracted across {num_pages} page(s).")

    # 2. Step 11A/11B Document Understanding
    doc_service = DocumentUnderstandingService()
    doc_understanding_res = doc_service.process_document(
        blocks=blocks, document_id="real_doc_qp1", page_sizes=page_sizes_dict
    )
    print(f"[*] Step 11A/11B Document Understanding Completed: {len(doc_understanding_res.regions)} regions, VLM Status: {doc_understanding_res.vlm_status}")

    # 3. Step 11C Intelligent Question Extraction
    extraction_service = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
    res: DocumentQuestionExtractionResult = extraction_service.extract_validated_questions(
        blocks=blocks,
        document_id="real_doc_qp1",
        doc_understanding_result=doc_understanding_res,
        page_sizes=page_sizes_dict,
    )

    print("\n" + "=" * 110)
    print("REGION-LEVEL STRUCTURE & EXTRACTION DIAGNOSTIC TABLE")
    print("=" * 110)
    print(f"{'Region ID':<12} | {'OCR Text (Preview)':<40} | {'11A Type':<15} | {'11B State':<12} | {'Final Question ID':<18}")
    print("-" * 110)

    for reg in doc_understanding_res.regions:
        txt_preview = reg.text.strip().replace("\n", " ")[:38]
        q_assoc = "REJECTED/NON-Q"
        for q in res.questions:
            if reg.region_id in q.source_region_ids:
                q_assoc = f"{q.id} ({q.question_type})"
                break
        print(f"{reg.region_id:<12} | {txt_preview:<40} | {reg.region_type:<15} | {reg.verification_state:<12} | {q_assoc:<18}")

    print("\n" + "=" * 110)
    print("VALIDATED QUESTION EXTRACTION RESULT SUMMARY")
    print("=" * 110)
    print(f"{'Order':<6} | {'Number':<10} | {'Type':<12} | {'Text (Preview)':<45} | {'Options':<8} | {'Pages':<6} | {'Confidence':<10}")
    print("-" * 110)

    for q in res.questions:
        txt_p = q.text.strip().replace("\n", " ")[:43]
        pages_str = ",".join(str(r.page) for r in q.source_regions)
        opts_count = len(q.extracted_options)
        print(f"{q.order_index:<6} | {q.number:<10} | {q.question_type:<12} | {txt_p:<45} | {opts_count:<8} | {pages_str:<6} | {q.extraction_confidence:<10.4f}")

    print("\n" + "=" * 110)
    print("DOCUMENTQUESTIONEXTRACTIONRESULT & EXTRACTIONAUDIT SUMMARY")
    print("=" * 110)
    audit = res.audit
    print(f"  Document ID:                    {res.document_id}")
    print(f"  Fallback Used:                  {res.fallback_used}")
    print(f"  Candidate Regions Count:        {audit.candidate_count}")
    print(f"  Accepted Questions Count:       {audit.accepted_question_count}")
    print(f"  Rejected Candidates Count:      {audit.rejected_count}")
    print(f"  Uncertain Candidates Count:     {audit.uncertain_count}")
    print(f"  Extracted Options Count:        {audit.option_count}")
    print(f"  Extracted Sections Count:       {audit.section_count}")
    print(f"  Multi-Region Questions Count:   {audit.multi_region_question_count}")
    print(f"  Multi-Page Questions Count:     {audit.multi_page_question_count}")
    
    if audit.rejection_reasons:
        print("\n  REJECTION REASONS BREAKDOWN:")
        for r_rec in audit.rejection_reasons[:5]:
            print(f"   - Region '{r_rec.region_id}' ({r_rec.classification}): '{r_rec.ocr_text[:40]}' -> Reason: {r_rec.reason}")

    print("=" * 110)
    print("REAL DOCUMENT DIAGNOSTIC COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_real_document_verification()
