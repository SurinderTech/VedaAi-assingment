#!/usr/bin/env python
"""
Inspect actual graph edges and relationships from real PDF
to verify if subquestion_of vs option_of bug exists
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService


def main():
    pdf_path = Path("/Users/surin/VedaAi-assingment/backend/scratch/test_corpus/multi_page_paper.pdf")
    
    if not pdf_path.exists():
        print(f"✗ PDF not found: {pdf_path}")
        return
    
    print("\n" + "="*80)
    print("ACTUAL GRAPH EDGES AND RELATIONSHIP TYPES")
    print("="*80)
    
    blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf")
    
    doc_service = DocumentUnderstandingService()
    result = doc_service.process_document(
        blocks=blocks,
        document_id="multi_page",
        page_sizes={i: [float(s[0]), float(s[1])] for i, s in enumerate(page_sizes, 1)},
        page_images=page_images,
        force_vlm_verification=True
    )
    
    print(f"\nGraph Edges: {len(result.structure_graph.edges)}")
    print("\nDetailed Edge List:")
    print("-" * 80)
    
    for i, edge in enumerate(result.structure_graph.edges, 1):
        source_node = result.structure_graph.nodes.get(edge.source_id)
        target_node = result.structure_graph.nodes.get(edge.target_id)
        
        source_role = source_node.role if source_node else "?"
        target_role = target_node.role if target_node else "?"
        source_text = (source_node.text[:30] if source_node and source_node.text else "?").replace("\n", " ")
        target_text = (target_node.text[:30] if target_node and target_node.text else "?").replace("\n", " ")
        
        print(f"\n[{i}] Relationship Type: {edge.relationship}")
        print(f"    Source: {edge.source_id[:20]}")
        print(f"      Role: {source_role}")
        print(f"      Text: {source_text}")
        print(f"    Target: {edge.target_id[:20]}")
        print(f"      Role: {target_role}")
        print(f"      Text: {target_text}")
        print(f"    Confidence: {edge.confidence}")
        print(f"    Evidence: {edge.evidence_sources}")
    
    # Check for suspicious relationship combinations
    print("\n\n" + "="*80)
    print("RELATIONSHIP SEMANTIC AUDIT")
    print("="*80)
    
    suspicious = []
    for edge in result.structure_graph.edges:
        source_node = result.structure_graph.nodes.get(edge.source_id)
        target_node = result.structure_graph.nodes.get(edge.target_id)
        
        if not source_node or not target_node:
            continue
        
        source_role = source_node.role
        target_role = target_node.role
        rel_type = edge.relationship
        
        # Check for semantic mismatches
        if source_role == "SUBQUESTION" and rel_type == "option_of":
            suspicious.append((edge, "SUBQUESTION should use 'subquestion_of', not 'option_of'"))
        elif source_role == "OPTION" and rel_type == "subquestion_of":
            suspicious.append((edge, "OPTION should use 'option_of', not 'subquestion_of'"))
    
    if suspicious:
        print("\n⚠️  SUSPICIOUS RELATIONSHIPS FOUND:")
        for edge, issue in suspicious:
            print(f"  - {issue}")
            print(f"    Edge: {edge.source_id[:20]} --[{edge.relationship}]--> {edge.target_id[:20]}")
    else:
        print("\n✓ No suspicious relationship mismatches found")
        print("  All relationship types appear semantically consistent with node roles")
    
    # Summary
    rel_counts = {}
    for edge in result.structure_graph.edges:
        rel_type = edge.relationship
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    print(f"\n\nRelationship Type Summary:")
    for rel_type, count in sorted(rel_counts.items()):
        print(f"  {rel_type}: {count}")


if __name__ == "__main__":
    main()
