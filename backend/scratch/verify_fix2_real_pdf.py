#!/usr/bin/env python
"""
FIX #2 REAL PDF VERIFICATION - ACTUAL QUESTION PAPER

Process the actual multi_page_paper.pdf and verify:
1. Real Q1 + 1(a)-1(j) structure in actual PDF
2. Actual graph relationships (subquestion_of vs option_of)
3. Actual grounding results on real document
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


def main():
    # Use actual test PDF
    pdf_path = Path("/Users/surin/VedaAi-assingment/backend/scratch/test_corpus/multi_page_paper.pdf")
    
    if not pdf_path.exists():
        print(f"✗ PDF not found: {pdf_path}")
        return
    
    print("\n" + "="*80)
    print("FIX #2 REAL PDF VERIFICATION - ACTUAL QUESTION PAPER")
    print("="*80)
    print(f"\nPDF: {pdf_path.name}")
    print(f"Size: {pdf_path.stat().st_size} bytes")
    
    # Process document
    print("\n[STEP 1] Processing PDF...")
    blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf")
    
    print(f"  Pages: {num_pages}")
    print(f"  Total OCR blocks: {len(blocks)}")
    
    for page in range(1, num_pages + 1):
        page_blocks = [b for b in blocks if b.page == page]
        print(f"  Page {page}: {len(page_blocks)} blocks")
    
    # Document understanding with VLM
    print("\n[STEP 2] Document Understanding (VLM enabled)...")
    doc_service = DocumentUnderstandingService()
    result = doc_service.process_document(
        blocks=blocks,
        document_id="multi_page",
        page_sizes={i: [float(s[0]), float(s[1])] for i, s in enumerate(page_sizes, 1)},
        page_images=page_images,
        force_vlm_verification=True
    )
    
    print(f"  VLM Status: {result.vlm_status}")
    print(f"  Total Regions: {len(result.regions)}")
    print(f"  Document Purpose: {result.document_purpose}")
    
    # Show Page 1 details in detail
    if result.vlm_page_understandings:
        p1_understanding = result.vlm_page_understandings[0]
        
        print(f"\n[STEP 3] PAGE 1 VLM STRUCTURES (Real from PDF):")
        print("-" * 76)
        print(f"  Image dimensions: {p1_understanding.image_dimensions}")
        print(f"  OCR blocks sent: {p1_understanding.ocr_blocks_sent}")
        print(f"  Structures found: {len(p1_understanding.structures)}")
        print(f"  Relationships found: {len(p1_understanding.relationships)}")
        print(f"  VLM provider/model: {p1_understanding.vlm_provider}/{p1_understanding.vlm_model}")
        print(f"  VLM result: {p1_understanding.vlm_result}")
        
        print(f"\n  STRUCTURES:")
        for i, struct in enumerate(p1_understanding.structures[:15], 1):
            print(f"\n    [{i}] Role: {struct.role}")
            if struct.display_number:
                print(f"        Display #: {struct.display_number}")
            if struct.display_label:
                print(f"        Label: {struct.display_label}")
            if struct.bbox:
                print(f"        BBox: [{struct.bbox.x:.0f}, {struct.bbox.y:.0f}, {struct.bbox.width:.0f}, {struct.bbox.height:.0f}]")
            print(f"        Region IDs: {struct.region_ids}")
            print(f"        Grounded IDs: {struct.grounded_region_ids}")
            print(f"        Status: {struct.grounding_status}")
            if struct.grounded_text:
                text_preview = struct.grounded_text[:60].replace("\n", " ")
                print(f"        Grounded Text: '{text_preview}'")
        
        if len(p1_understanding.structures) > 15:
            print(f"\n    ... and {len(p1_understanding.structures) - 15} more structures")
        
        print(f"\n  RELATIONSHIPS:")
        for i, rel in enumerate(p1_understanding.relationships[:10], 1):
            print(f"    [{i}] {rel.source_ids} --[{rel.relationship_type}]--> {rel.target_ids}")
    
    # Extract questions
    print(f"\n[STEP 4] QUESTION EXTRACTION:")
    print("-" * 76)
    extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
    extraction = extractor.extract_validated_questions(
        blocks=blocks,
        document_id="multi_page",
        doc_understanding_result=result
    )
    
    print(f"  Questions Extracted: {len(extraction.questions)}")
    
    # Show Q1 and all its subquestions
    q1_list = [q for q in extraction.questions if q.number == "1"]
    if q1_list:
        main_q1 = q1_list[0]
        print(f"\n  Q1: {main_q1.text[:60]}")
        print(f"      Page: {main_q1.page}")
        print(f"      Type: {main_q1.question_type}")
        print(f"      Parent: {main_q1.parent_question_id}")
        print(f"      Options: {len(main_q1.extracted_options)}")
        
        # Find subquestions
        subquestions = [q for q in extraction.questions if q.parent_question_id == main_q1.id]
        print(f"      Subquestions: {len(subquestions)}")
        
        for sq in sorted(subquestions, key=lambda x: x.number if x.number else ""):
            print(f"        {sq.number}: {sq.text[:50]}")
            print(f"               ID: {sq.id}")
            print(f"               Type: {sq.question_type}")
    
    # Show actual graph structure
    print(f"\n[STEP 5] ACTUAL STRUCTURE GRAPH:")
    print("-" * 76)
    
    if result.structure_graph:
        print(f"  Total Nodes: {len(result.structure_graph.nodes)}")
        print(f"  Total Edges: {len(result.structure_graph.edges)}")
        
        # Find Q1 node
        q1_nodes = [node for nid, node in result.structure_graph.nodes.items() if node.role == "QUESTION" and "1" in (node.text or "")]
        
        if q1_nodes:
            print(f"\n  Q1 Node:")
            q1_node = q1_nodes[0]
            print(f"    ID: {q1_node.region_id}")
            print(f"    Role: {q1_node.role}")
            print(f"    Text: {q1_node.text[:60]}")
            print(f"    Page: {q1_node.page}")
            
            # Find edges connected to Q1
            incoming = [e for e in result.structure_graph.edges if e.target_id == q1_node.region_id]
            outgoing = [e for e in result.structure_graph.edges if e.source_id == q1_node.region_id]
            
            print(f"\n  Edges connected to Q1:")
            print(f"    Incoming ({len(incoming)}):")
            for edge in incoming[:10]:
                print(f"      {edge.source_id[:20]} --[{edge.relationship}]--> Q1")
            
            print(f"    Outgoing ({len(outgoing)}):")
            for edge in outgoing[:10]:
                print(f"      Q1 --[{edge.relationship}]--> {edge.target_id[:20]}")
            
            if len(outgoing) > 10:
                print(f"      ... and {len(outgoing) - 10} more")
    
    print(f"\n✓ Real PDF Diagnostic Complete")


if __name__ == "__main__":
    main()
