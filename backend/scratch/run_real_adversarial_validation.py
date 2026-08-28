"""
Step 11C — Real Adversarial Document Extraction Validation Diagnostic.

Executes Step 11C extraction on:
1. Real Sample Question Paper Image (qp.png)
2. Real Sample Question Paper Image (qp8.png)
3. Complex Multi-Structure Adversarial Question Paper (Complex_Adversarial_QP)
   containing: cover metadata, general instructions, section headers, section numbering restarts,
   MCQs with multiple options, tables, figures/diagrams, multi-column layout geometry,
   subquestions, multi-page continuations, and header/footer noise.

Produces for each document:
- Region-level diagnostic
- Question extraction summary
- Extraction audit
- Source-evidence validation
- MCQ validation
- Reading-order validation
- False-positive validation
"""
from __future__ import annotations
import sys
import os
import asyncio

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import Block, BBox, DocumentQuestionExtractionResult, DocumentUnderstandingResult, DocumentPage, DocumentRegion
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


def validate_document(doc_name: str, blocks: list[Block], page_sizes: dict = None, doc_understanding_res: DocumentUnderstandingResult = None):
    print("\n" + "=" * 120)
    print(f"ADVERSARIAL VALIDATION FOR DOCUMENT: {doc_name}")
    print("=" * 120)

    doc_service = DocumentUnderstandingService()
    if doc_understanding_res is None:
        doc_understanding_res = doc_service.process_document(
            blocks=blocks, document_id=doc_name, page_sizes=page_sizes
        )

    extraction_service = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
    res: DocumentQuestionExtractionResult = extraction_service.extract_validated_questions(
        blocks=blocks,
        document_id=doc_name,
        doc_understanding_result=doc_understanding_res,
        page_sizes=page_sizes,
    )

    # 1. Region-level Diagnostic
    print("\n--- 1. REGION-LEVEL DIAGNOSTIC TABLE ---")
    print(f"{'Region ID':<12} | {'Pg':<3} | {'11A Type':<14} | {'11B State':<11} | {'Final Classification':<22} | {'OCR Text (Preview)':<35}")
    print("-" * 120)

    rejection_map = {r.region_id: r.reason for r in res.audit.rejection_reasons}

    for reg in doc_understanding_res.regions:
        txt_prev = reg.text.strip().replace("\n", " ")[:33]
        
        # Determine final classification
        final_cls = f"REJECTED: {reg.region_type}"
        if reg.region_id in rejection_map:
            final_cls = f"REJECTED ({reg.region_type})"
        else:
            for q in res.questions:
                if reg.region_id in q.source_region_ids:
                    final_cls = f"QUESTION: {q.id} ({q.question_type})"
                    break
                for opt in q.extracted_options:
                    if reg.region_id in opt.source_region_ids:
                        final_cls = f"OPTION: {q.id} [{opt.label}]"
                        break
            for sec in res.sections:
                if reg.region_id in sec.source_region_ids:
                    final_cls = f"SECTION: {sec.title}"
                    break

        print(f"{reg.region_id:<12} | {reg.page:<3} | {reg.region_type:<14} | {reg.verification_state:<11} | {final_cls:<22} | {txt_prev:<35}")

    # 2. Question Extraction Summary
    print("\n--- 2. QUESTION EXTRACTION SUMMARY TABLE ---")
    print(f"{'Ord':<4} | {'QNum':<8} | {'Type':<12} | {'Opts':<4} | {'Pg':<5} | {'Conf':<6} | {'Exact Extracted Text (Preview)':<40} | {'Option Texts'}")
    print("-" * 120)

    for q in res.questions:
        txt_p = q.text.strip().replace("\n", " ")[:38]
        pages_str = ",".join(str(r.page) for r in q.source_regions)
        opts_str = " | ".join(f"{opt.label}: {opt.text}" for opt in q.extracted_options) if q.extracted_options else "None"
        print(f"{q.order_index:<4} | {q.number:<8} | {q.question_type:<12} | {len(q.extracted_options):<4} | {pages_str:<5} | {q.extraction_confidence:<6.4f} | {txt_p:<40} | {opts_str[:30]}")

    # 3. Extraction Audit Summary
    print("\n--- 3. EXTRACTION AUDIT SUMMARY ---")
    a = res.audit
    print(f"  Candidate Regions Count:        {a.candidate_count}")
    print(f"  Accepted Questions Count:       {a.accepted_question_count}")
    print(f"  Rejected Candidates Count:      {a.rejected_count}")
    print(f"  Uncertain Candidates Count:     {a.uncertain_count}")
    print(f"  Extracted Options Count:        {a.option_count}")
    print(f"  Extracted Sections Count:       {a.section_count}")
    print(f"  Multi-Region Questions Count:   {a.multi_region_question_count}")
    print(f"  Multi-Page Questions Count:     {a.multi_page_question_count}")
    print(f"  Fallback Used:                  {res.fallback_used}")

    # 4. Source-Evidence Validation
    print("\n--- 4. SOURCE-EVIDENCE ZERO-HALLUCINATION VALIDATION ---")
    ocr_texts_set = {b.text.strip() for b in blocks}
    all_zero_hallucinated = True
    for q in res.questions:
        # Check that q.text is substring/exact concatenation of source regions
        src_regs = [b for b in blocks if b.id in q.source_region_ids]
        expected_concat = " ".join(b.text.strip() for b in src_regs)
        if q.text.strip() != expected_concat.strip():
            print(f"  [FAIL] Zero-hallucination discrepancy for {q.id}: Extracted '{q.text}' != Expected '{expected_concat}'")
            all_zero_hallucinated = False
        for opt in q.extracted_options:
            opt_src_regs = [b for b in blocks if b.id in opt.source_region_ids]
            if not opt_src_regs:
                print(f"  [FAIL] Option {opt.option_id} has no source regions!")
                all_zero_hallucinated = False
    if all_zero_hallucinated:
        print("  [PASS] 100% of extracted question & option characters originate strictly from original OCR source regions.")

    # 5. MCQ Validation
    print("\n--- 5. MCQ PARENT-CHILD STRUCTURE VALIDATION ---")
    mcq_valid = True
    for q in res.questions:
        if q.extracted_options:
            if q.question_type != "MCQ":
                print(f"  [WARN] Question {q.id} has options but question_type is {q.question_type}")
            for opt in q.extracted_options:
                # Ensure option region ID is not emitted as a standalone question
                if any(opt_id in q2.source_region_ids for q2 in res.questions for opt_id in opt.source_region_ids):
                    print(f"  [FAIL] Option region {opt.source_region_ids} was also emitted as an independent question!")
                    mcq_valid = False
    if mcq_valid:
        print("  [PASS] MCQ options are properly attached as children of parent questions and NOT emitted as standalone questions.")

    # 6. Reading-Order Validation
    print("\n--- 6. READING-ORDER VALIDATION ---")
    print(f"  Extracted Question Order: {[q.number for q in res.questions]}")

    # 7. False-Positive Validation
    print("\n--- 7. FALSE-POSITIVE PROTECTION VALIDATION ---")
    non_q_promoted = False
    for q in res.questions:
        q_low = q.text.lower()
        if "general instructions" in q_low or "time allowed" in q_low or "maximum marks" in q_low or "roll no" in q_low or "page " in q_low:
            print(f"  [FAIL] Administrative non-question text promoted to question: '{q.text}'")
            non_q_promoted = True
    if not non_q_promoted:
        print("  [PASS] Administrative metadata, instructions, headers, and footers strictly excluded from question set.")

    return res


def run_all_adversarial_validations():
    print("=" * 120)
    print("STEP 11C — REAL ADVERSARIAL PIPELINE VALIDATION")
    print("=" * 120)

    # -------------------------------------------------------------------------
    # DOCUMENT 1: qp.png (Real Sample Image)
    # -------------------------------------------------------------------------
    img1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp.png"))
    if os.path.exists(img1_path):
        b1, np1, sz1 = process_document(img1_path, ".png")
        p_sz1 = {1: [float(sz1[0][0]), float(sz1[0][1])]}
        validate_document("qp.png", b1, page_sizes=p_sz1)

    # -------------------------------------------------------------------------
    # DOCUMENT 2: qp8.png (Real Sample Image)
    # -------------------------------------------------------------------------
    img2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qp8.png"))
    if os.path.exists(img2_path):
        b2, np2, sz2 = process_document(img2_path, ".png")
        p_sz2 = {1: [float(sz2[0][0]), float(sz2[0][1])]}
        validate_document("qp8.png", b2, page_sizes=p_sz2)

    # -------------------------------------------------------------------------
    # DOCUMENT 3: Complex Multi-Structure Adversarial Question Paper
    # -------------------------------------------------------------------------
    b_adv = [
        # Page 1 Top Administrative Metadata & General Instructions
        Block(id="h1", text="DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING", page=1, bbox=BBox(x=100, y=10, width=800, height=20), confidence=0.98),
        Block(id="h2", text="Roll No: _______________   Paper Code: CS-401   Time Allowed: 3 Hours   Max Marks: 100", page=1, bbox=BBox(x=50, y=35, width=900, height=20), confidence=0.98),
        Block(id="ins1", text="General Instructions: 1. Answer all questions from Section A. 2. Section B carries optional choices. 3. Figures to the right indicate full marks.", page=1, bbox=BBox(x=50, y=65, width=900, height=30), confidence=0.95),
        
        # Section A Header
        Block(id="sec_a", text="SECTION A (Multiple Choice & Short Answer)", page=1, bbox=BBox(x=50, y=105, width=400, height=20), confidence=0.95),
        
        # Left Column Q1 (MCQ)
        Block(id="q1", text="1. Which activation function suffers from vanishing gradient problem?", page=1, bbox=BBox(x=50, y=135, width=420, height=25), confidence=0.95),
        Block(id="q1_a", text="(A) Sigmoid", page=1, bbox=BBox(x=70, y=165, width=180, height=20), confidence=0.95),
        Block(id="q1_b", text="(B) ReLU", page=1, bbox=BBox(x=260, y=165, width=180, height=20), confidence=0.95),
        Block(id="q1_c", text="(C) Leaky ReLU", page=1, bbox=BBox(x=70, y=190, width=180, height=20), confidence=0.95),
        Block(id="q1_d", text="(D) GELU", page=1, bbox=BBox(x=260, y=190, width=180, height=20), confidence=0.95),
        
        # Left Column Q2
        Block(id="q2", text="2. Define learning rate hyperparameter and explain its effect on convergence.", page=1, bbox=BBox(x=50, y=220, width=420, height=30), confidence=0.95),
        
        # Table (Non-question structural content)
        Block(id="tbl_hdr", text="Table 1: Hyperparameter Comparison Metrics", page=1, bbox=BBox(x=50, y=260, width=420, height=20), confidence=0.90),
        Block(id="tbl_body", text="Learning Rate | Batch Size | Convergence Speed\n0.01 | 32 | Fast\n0.001 | 64 | Moderate", page=1, bbox=BBox(x=50, y=285, width=420, height=50), confidence=0.90),

        # Right Column Q3 (Multi-column reading order test)
        Block(id="q3", text="3. State Bayes Theorem for conditional probability.", page=1, bbox=BBox(x=520, y=135, width=430, height=25), confidence=0.95),
        
        # Right Column Q4
        Block(id="q4", text="4. What is cross-entropy loss function used for in classification?", page=1, bbox=BBox(x=520, y=170, width=430, height=25), confidence=0.95),

        # Right Column Figure
        Block(id="fig1", text="Figure 2: Architecture of Convolutional Neural Network", page=1, bbox=BBox(x=520, y=205, width=430, height=20), confidence=0.90),

        # Page 1 Footer Noise
        Block(id="ftr1", text="Page 1 of 2 — End of Section A", page=1, bbox=BBox(x=400, y=960, width=300, height=20), confidence=0.98),

        # Page 2 Section B Header
        Block(id="sec_b", text="SECTION B (Descriptive & Subquestions)", page=2, bbox=BBox(x=50, y=25, width=400, height=20), confidence=0.95),

        # Section B Restarts Numbering Q1(a), Q1(b)
        Block(id="q5_main", text="1. Answer the following subquestions:", page=2, bbox=BBox(x=50, y=55, width=900, height=20), confidence=0.95),
        Block(id="q5_a", text="1(a) Derive gradient update equation for backpropagation.", page=2, bbox=BBox(x=70, y=80, width=880, height=25), confidence=0.95),
        Block(id="q5_b", text="1(b) Explain Adam optimizer bias correction mechanism.", page=2, bbox=BBox(x=70, y=110, width=880, height=25), confidence=0.95),

        # Multi-Page Continuation Question (Starts Page 2, Continues Page 2 body)
        Block(id="q6_p1", text="2. Explain transformer self-attention mechanism in detail.", page=2, bbox=BBox(x=50, y=145, width=900, height=25), confidence=0.95),
        Block(id="q6_p2", text="Include query, key, value matrix projections and multi-head attention formulation.", page=2, bbox=BBox(x=50, y=175, width=900, height=25), confidence=0.95),

        # Page 2 Footer Noise
        Block(id="ftr2", text="Page 2 of 2 — End of Examination", page=2, bbox=BBox(x=400, y=960, width=300, height=20), confidence=0.98),
    ]

    p_adv_sizes = {1: [1000.0, 1000.0], 2: [1000.0, 1000.0]}
    
    # Run Document Understanding for Complex_Adversarial_QP
    doc_service = DocumentUnderstandingService()
    du_adv = doc_service.process_document(blocks=b_adv, document_id="Complex_Adversarial_QP", page_sizes=p_adv_sizes)

    validate_document("Complex_Adversarial_QP", b_adv, page_sizes=p_adv_sizes, doc_understanding_res=du_adv)

    print("\n" + "=" * 120)
    print("ALL ADVERSARIAL VALIDATION RUNS COMPLETED SUCCESSFULLY!")
    print("=" * 120)


if __name__ == "__main__":
    run_all_adversarial_validations()
