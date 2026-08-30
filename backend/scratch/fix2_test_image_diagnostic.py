#!/usr/bin/env python
"""
TEST IMAGE DIAGNOSTIC - Detailed Grounding Verification

This processes qp.png to show detailed grounding behavior.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


def main():
    qp_path = Path("/Users/surin/VedaAi-assingment/backend/scratch/qp.png")
    
    if not qp_path.exists():
        print(f"Error: {qp_path} not found")
        return
    
    print("\n" + "="*80)
    print("FIX #2 TEST IMAGE DIAGNOSTIC")
    print("="*80)
    print(f"\nProcessing: {qp_path.name}")
    
    # Step 1: Process document
    print("\n[STEP 1] Document Processing")
    print("-" * 60)
    blocks, num_pages, page_sizes, page_images = process_document(str(qp_path), ".png")
    
    print(f"  Pages: {num_pages}")
    print(f"  Page Size: {page_sizes[0]}")
    print(f"  OCR Blocks: {len(blocks)}")
    
    for b in blocks:
        print(f"    - [{b.id:8s}] {b.text:40s} bbox=[{b.bbox.x:.0f}, {b.bbox.y:.0f}, {b.bbox.width:.0f}, {b.bbox.height:.0f}]")
    
    # Step 2: Document Understanding with VLM
    print("\n[STEP 2] Document Understanding Service")
    print("-" * 60)
    doc_service = DocumentUnderstandingService()
    result = doc_service.process_document(
        blocks=blocks,
        document_id="qp_test",
        page_sizes={1: [float(page_sizes[0][0]), float(page_sizes[0][1])]},
        page_images=page_images,
        force_vlm_verification=True
    )
    
    print(f"  VLM Status: {result.vlm_status}")
    print(f"  Total Regions: {len(result.regions)}")
    print(f"  Document Purpose: {result.document_purpose}")
    
    # Step 3: Page 1 Diagnostic
    if result.vlm_page_understandings:
        understanding = result.vlm_page_understandings[0]
        
        print(f"\n[STEP 3] Page 1 VLM Understanding")
        print("-" * 60)
        print(f"  Image Dimensions: {understanding.image_dimensions}")
        print(f"  OCR Blocks Sent: {understanding.ocr_blocks_sent}")
        print(f"  VLM Provider: {understanding.vlm_provider}/{understanding.vlm_model}")
        print(f"  VLM Result: {understanding.vlm_result}")
        print(f"  Finish Reason: {understanding.finish_reason}")
        print(f"  Structure Source: {understanding.structure_source}")
        print(f"  Structures Found: {len(understanding.structures)}")
        
        # Show each structure with grounding details
        print(f"\n[STEP 4] Page 1 Structures with Grounding Details")
        print("-" * 60)
        
        for i, struct in enumerate(understanding.structures, 1):
            print(f"\n  Structure {i}:")
            print(f"    Role: {struct.role}")
            if struct.display_number:
                print(f"    Display #: {struct.display_number}")
            if struct.display_label:
                print(f"    Label: {struct.display_label}")
            
            if struct.bbox:
                print(f"    Visual BBox: x={struct.bbox.x:.0f}, y={struct.bbox.y:.0f}, w={struct.bbox.width:.0f}, h={struct.bbox.height:.0f}")
            
            print(f"    Region IDs (direct OCR): {struct.region_ids}")
            print(f"    Grounded Region IDs: {struct.grounded_region_ids}")
            print(f"    Grounding Status: {struct.grounding_status}")
            
            if struct.grounded_region_ids:
                print(f"    Grounded Text: '{struct.grounded_text}'")
            
            print(f"    Confidence: {struct.confidence:.2f}")
            print(f"    Reasoning: {struct.reasoning}")
    
    # Step 5: Question Extraction
    print(f"\n[STEP 5] Question Extraction")
    print("-" * 60)
    
    extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
    extraction = extractor.extract_validated_questions(
        blocks=blocks,
        document_id="qp_test",
        doc_understanding_result=result
    )
    
    print(f"  Questions Extracted: {len(extraction.questions)}")
    
    for q in extraction.questions:
        print(f"\n    Q{q.number}:")
        print(f"      Text: {q.text[:60]}")
        print(f"      Page: {q.page}")
        print(f"      Type: {q.question_type}")
        print(f"      Options: {len(q.extracted_options)}")
        for opt in q.extracted_options:
            print(f"        - {opt.label}: {opt.text[:40]}")
    
    # Step 6: Grounding Summary
    print(f"\n[STEP 6] Grounding Summary")
    print("-" * 60)
    
    if result.vlm_page_understandings:
        all_structs = result.vlm_page_understandings[0].structures
        
        grounded = sum(1 for s in all_structs if s.grounding_status == "GROUNDED")
        partial = sum(1 for s in all_structs if s.grounding_status == "PARTIALLY_GROUNDED")
        ungrounded = sum(1 for s in all_structs if s.grounding_status == "UNGROUNDED")
        
        print(f"  Total Structures: {len(all_structs)}")
        print(f"  Grounded: {grounded}")
        print(f"  Partially Grounded: {partial}")
        print(f"  Ungrounded: {ungrounded}")
        
        if len(all_structs) > 0:
            print(f"  Success Rate: {100*grounded/len(all_structs):.1f}%")
    
    print(f"\n✓ Diagnostic Complete")


if __name__ == "__main__":
    main()
