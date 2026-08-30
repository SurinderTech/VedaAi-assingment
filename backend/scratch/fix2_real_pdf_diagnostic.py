#!/usr/bin/env python
"""
REAL PDF DIAGNOSTIC FOR FIX #2 VERIFICATION

Processes the actual problematic question paper and shows:
1. Page-level VLM diagnostic (structure count, grounding status)
2. Page 1 visual hierarchy (Q1 with 1a-1j)
3. Grounding status for each structure
4. MCQ section behavior
5. Administrative document handling
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService

import json
from pathlib import Path


def find_real_pdfs():
    """Find the real question paper PDFs."""
    temp_dir = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads")
    qp_files = []
    
    if temp_dir.exists():
        qp_files = list(temp_dir.glob("*.pdf"))
    
    return qp_files


def run_pdf_diagnostic(pdf_path):
    """Run diagnostic on a single PDF."""
    print("\n" + "="*80)
    print(f"PDF: {pdf_path.name}")
    print("="*80)
    
    try:
        blocks, num_pages, page_sizes, page_images = process_document(
            str(pdf_path),
            ".pdf",
            force_ocr=False
        )
        
        print(f"\n📄 Document: {pdf_path.name}")
        print(f"   Pages: {num_pages}")
        print(f"   Total OCR blocks: {len(blocks)}")
        
        doc_service = DocumentUnderstandingService()
        result = doc_service.process_document(
            blocks=blocks,
            document_id=pdf_path.stem,
            page_sizes={i: [float(s[0]), float(s[1])] for i, s in enumerate(page_sizes, 1)},
            page_images=page_images,
            force_vlm_verification=True
        )
        
        print(f"\n🔍 Understanding Result:")
        print(f"   VLM Status: {result.vlm_status}")
        print(f"   Total Regions: {len(result.regions)}")
        print(f"   Total Relationships: {len(result.relationships)}")
        print(f"   Document Purpose: {result.document_purpose}")
        
        # Page-by-page diagnostics
        print(f"\n📊 Page-Level Diagnostics:")
        print("-" * 80)
        
        for understanding in result.vlm_page_understandings:
            page_num = understanding.page_number
            print(f"\nPage {page_num}:")
            print(f"  Image Dimensions: {understanding.image_dimensions}")
            print(f"  Image Bytes: {understanding.image_bytes}")
            print(f"  Image Sent: {understanding.image_sent}")
            print(f"  OCR Blocks: {understanding.ocr_blocks_sent}")
            print(f"  VLM Provider: {understanding.vlm_provider}")
            print(f"  VLM Model: {understanding.vlm_model}")
            print(f"  VLM Result: {understanding.vlm_result}")
            print(f"  Finish Reason: {understanding.finish_reason}")
            print(f"  Structure Source: {understanding.structure_source}")
            print(f"  Structures: {len(understanding.structures)}")
            print(f"  Relationships: {len(understanding.relationships)}")
            
            # Count grounding status
            grounded = 0
            partially = 0
            ungrounded = 0
            for struct in understanding.structures:
                if struct.grounding_status == "GROUNDED":
                    grounded += 1
                elif struct.grounding_status == "PARTIALLY_GROUNDED":
                    partially += 1
                elif struct.grounding_status == "UNGROUNDED":
                    ungrounded += 1
            
            print(f"  ├─ Grounded: {grounded}")
            print(f"  ├─ Partially Grounded: {partially}")
            print(f"  └─ Ungrounded: {ungrounded}")
            
            # Show structures for Page 1
            if page_num == 1:
                print(f"\n  📋 Page 1 Structures (Visual Hierarchy):")
                print("-" * 76)
                for i, struct in enumerate(understanding.structures, 1):
                    print(f"\n  [{i}] Role: {struct.role}")
                    if struct.display_number:
                        print(f"      Display #: {struct.display_number}")
                    if struct.display_label:
                        print(f"      Label: {struct.display_label}")
                    if struct.bbox:
                        print(f"      Visual BBox: [{struct.bbox.x}, {struct.bbox.y}, {struct.bbox.width}, {struct.bbox.height}]")
                    print(f"      Region IDs (OCR-grounded): {struct.region_ids}")
                    print(f"      Grounded Region IDs: {struct.grounded_region_ids}")
                    print(f"      Grounding Status: {struct.grounding_status}")
                    if struct.grounded_text:
                        text_preview = struct.grounded_text[:60].replace("\n", " ")
                        print(f"      Grounded Text: {text_preview}...")
                    print(f"      Confidence: {struct.confidence:.2f}")
                    print(f"      Reasoning: {struct.reasoning}")
        
        # Extract questions
        from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
        extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
        extraction_result = extractor.extract_validated_questions(
            blocks=blocks,
            document_id=pdf_path.stem,
            doc_understanding_result=result
        )
        
        print(f"\n❓ Question Extraction:")
        print(f"   Questions Found: {len(extraction_result.questions)}")
        
        for q in extraction_result.questions[:5]:  # Show first 5
            print(f"\n   Q{q.number}: {q.text[:60].replace(chr(10), ' ')}")
            print(f"      Page: {q.page}")
            print(f"      Options: {len(q.extracted_options)}")
            if q.parent_question_id:
                print(f"      Parent: {q.parent_question_id}")
        
        if len(extraction_result.questions) > 5:
            print(f"\n   ... and {len(extraction_result.questions) - 5} more")
        
        print(f"\n✓ PDF Diagnostic Complete")
        return result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*80)
    print("FIX #2 REAL PDF VERIFICATION")
    print("="*80)
    
    # Find real PDFs in temp directory
    pdf_files = find_real_pdfs()
    
    if not pdf_files:
        print("\n⚠ No PDFs found in temp directory")
        print("Looking for test PDFs in workspace...")
        
        # Try workspace test files
        scratch_dir = Path("/Users/surin/VedaAi-assingment/backend/scratch")
        if scratch_dir.exists():
            test_images = list(scratch_dir.glob("*.png"))
            if test_images:
                print(f"\nFound {len(test_images)} test images")
                for img in test_images[:3]:
                    print(f"  - {img.name}")
        
        print("\n⚠ Recommendation: Upload question paper PDF to verify Fix #2")
        return
    
    print(f"\nFound {len(pdf_files)} PDF(s) in temp directory")
    
    for pdf_path in sorted(pdf_files)[:3]:  # Process first 3
        result = run_pdf_diagnostic(pdf_path)
        
        if result and len(result.vlm_page_understandings) > 0:
            # Show summary
            all_grounded = sum(
                1 for u in result.vlm_page_understandings
                for s in u.structures
                if s.grounding_status == "GROUNDED"
            )
            all_ungrounded = sum(
                1 for u in result.vlm_page_understandings
                for s in u.structures
                if s.grounding_status == "UNGROUNDED"
            )
            
            total_structures = sum(
                len(u.structures)
                for u in result.vlm_page_understandings
            )
            
            print(f"\n📈 Summary:")
            print(f"   Total Structures: {total_structures}")
            print(f"   Grounded: {all_grounded}")
            print(f"   Ungrounded: {all_ungrounded}")
            print(f"   Success Rate: {100*all_grounded/(total_structures or 1):.1f}%")


if __name__ == "__main__":
    main()
