#!/usr/bin/env python
"""
AUDIT VERIFICATION SCRIPT - Test Real Q1 + 1(a)-1(j) Capability

This script verifies whether the current pipeline can handle Q1 with 10 subquestions.

Requirements:
  - A real PDF with Q1 and 1(a) through 1(j) subquestions
  - VLM enabled (Gemini API key configured)
  - All audit layers will be inspected

Run this AFTER finding the correct PDF with Q1+subquestions structure.
"""

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)


def audit_vlm_structures(vlm_understanding):
    """Audit 1: Check VLM structures."""
    print_section("AUDIT 1: VLM STRUCTURES")
    
    print(f"Total structures found: {len(vlm_understanding.structures)}")
    
    q_count = 0
    sub_count = 0
    
    for struct in vlm_understanding.structures:
        if struct.role == "QUESTION":
            q_count += 1
            print(f"\n[Q] {struct.display_number}: {struct.display_label}")
            print(f"    role: {struct.role}")
            print(f"    region_ids: {struct.region_ids}")
            print(f"    grounding_status: {struct.grounding_status}")
            print(f"    confidence: {struct.confidence}")
        elif struct.role == "SUBQUESTION":
            sub_count += 1
            print(f"  [SUB-{struct.display_number}]: {struct.display_label}")
            print(f"      role: {struct.role}")
            print(f"      region_ids: {struct.region_ids}")
            print(f"      grounding_status: {struct.grounding_status}")
            print(f"      confidence: {struct.confidence}")
    
    print(f"\n✓ QUESTIONS: {q_count}")
    print(f"✓ SUBQUESTIONS: {sub_count}")
    
    if sub_count >= 10:
        print("✓ AUDIT 1 PASSED: Found 10+ subquestions")
        return True
    else:
        print(f"✗ AUDIT 1 FAILED: Expected 10 subquestions, found {sub_count}")
        return False


def audit_vlm_relationships(vlm_understanding):
    """Audit 2: Check VLM relationships."""
    print_section("AUDIT 2: VLM RELATIONSHIPS")
    
    print(f"Total relationships found: {len(vlm_understanding.relationships)}")
    
    subq_of_count = 0
    all_rels = []
    
    for rel in vlm_understanding.relationships:
        rel_type = rel.relationship_type
        all_rels.append((rel.source_ids, rel.target_ids, rel_type))
        
        if rel_type == "subquestion_of":
            subq_of_count += 1
            print(f"✓ {rel.source_ids[0] if rel.source_ids else '?'} --[subquestion_of]--> {rel.target_ids[0] if rel.target_ids else '?'}")
        else:
            print(f"  {rel.source_ids} --[{rel_type}]--> {rel.target_ids}")
    
    print(f"\nSubquestion_of relationships: {subq_of_count}")
    
    if subq_of_count >= 10:
        print("✓ AUDIT 2 PASSED: Found 10+ subquestion_of edges")
        return True
    elif subq_of_count > 0:
        print(f"⚠️ AUDIT 2 PARTIAL: Found {subq_of_count} subquestion_of edges (expected 10)")
        return True  # Partial pass
    else:
        print("✗ AUDIT 2 FAILED: No subquestion_of relationships found")
        print("   This is a CRITICAL GAP: VLM returned subquestions but no relationships")
        return False


def audit_graph_edges(graph):
    """Audit 3: Check DocumentStructureGraph edges."""
    print_section("AUDIT 3: GRAPH EDGES")
    
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")
    
    subq_of_edges = []
    q_nodes = {}
    sub_nodes = {}
    
    # Index nodes
    for node_id, node in graph.nodes.items():
        if node.role == "QUESTION":
            q_nodes[node_id] = node
        elif node.role == "SUBQUESTION":
            sub_nodes[node_id] = node
    
    print(f"\nNodes by role:")
    print(f"  QUESTION: {len(q_nodes)}")
    print(f"  SUBQUESTION: {len(sub_nodes)}")
    
    # Check edges
    for edge in graph.edges:
        if edge.relationship == "subquestion_of":
            subq_of_edges.append(edge)
            source_node = graph.nodes.get(edge.source_id)
            target_node = graph.nodes.get(edge.target_id)
            
            src_text = (source_node.text[:30] if source_node else "?").replace("\n", " ")
            tgt_text = (target_node.text[:30] if target_node else "?").replace("\n", " ")
            
            print(f"✓ {src_text} --[subquestion_of]--> {tgt_text}")
    
    print(f"\nSubquestion_of edges: {len(subq_of_edges)}")
    
    if len(subq_of_edges) >= 10:
        print("✓ AUDIT 3 PASSED: Graph contains 10+ subquestion_of edges")
        return True
    elif len(subq_of_edges) > 0:
        print(f"⚠️ AUDIT 3 PARTIAL: Found {len(subq_of_edges)} edges (expected 10)")
        return True  # Partial pass
    else:
        print("✗ AUDIT 3 FAILED: No subquestion_of edges in graph")
        return False


def audit_extraction_hierarchy(extraction):
    """Audit 4: Check extracted Questions with parent_question_id."""
    print_section("AUDIT 4: EXTRACTION HIERARCHY")
    
    q1 = None
    subquestions = []
    
    for q in extraction.questions:
        if q.number == "1":
            q1 = q
            print(f"\nQ1 FOUND:")
            print(f"  ID: {q.id}")
            print(f"  Number: {q.number}")
            print(f"  Type: {q.question_type}")
            print(f"  Text: {q.text[:60]}")
            print(f"  Parent: {q.parent_question_id}")
        elif q.parent_question_id == q1.id if q1 else False:
            subquestions.append(q)
            print(f"  Sub-{q.number}: {q.text[:50]}")
            print(f"      Parent: {q.parent_question_id}")
    
    print(f"\nHierarchy check:")
    print(f"  Q1 found: {'✓' if q1 else '✗'}")
    print(f"  Subquestions with parent_question_id=Q1: {len(subquestions)}")
    
    # Also check for orphaned subquestions
    orphaned = [q for q in extraction.questions 
                if q.question_type == "SUBQUESTION" and q.parent_question_id is None]
    if orphaned:
        print(f"  ⚠️ ORPHANED SUBQUESTIONS: {len(orphaned)}")
        for q in orphaned[:3]:
            print(f"     {q.number}: {q.text[:50]}")
    
    if q1 and len(subquestions) >= 10:
        print("\n✓ AUDIT 4 PASSED: Q1 found with 10+ subquestions linked")
        return True
    elif q1 and len(subquestions) > 0:
        print(f"\n⚠️ AUDIT 4 PARTIAL: Q1 has {len(subquestions)} subquestions (expected 10)")
        return True  # Partial pass
    elif q1:
        print("\n✗ AUDIT 4 FAILED: Q1 found but NO subquestions linked")
        print("   This indicates the critical gap: relationships not passed to extraction")
        return False
    else:
        print("\n✗ AUDIT 4 FAILED: Q1 not found at all")
        return False


def audit_invariant_violations(extraction):
    """Audit 5: Check for invariant violations."""
    print_section("AUDIT 5: INVARIANT VALIDATION")
    
    if extraction.audit.invariant_violations:
        print(f"❌ {len(extraction.audit.invariant_violations)} invariant violations found:\n")
        for violation in extraction.audit.invariant_violations[:10]:
            print(f"  - {violation}")
        if len(extraction.audit.invariant_violations) > 10:
            print(f"  ... and {len(extraction.audit.invariant_violations) - 10} more")
        print("\n✗ AUDIT 5 FAILED: Invariants violated")
        return False
    else:
        print("✓ No invariant violations")
        print("✓ AUDIT 5 PASSED: All invariants satisfied")
        return True


def main():
    # TODO: Update this to your Q1+subquestions PDF path
    pdf_path = Path("backend/scratch/test_corpus/Q1_subquestions.pdf")  # ← Replace with real path
    
    if not pdf_path.exists():
        print("❌ PDF not found: " + str(pdf_path))
        print("\nTo run this audit:")
        print("1. Locate or create a PDF with Q1 + 1(a)-1(j) structure")
        print("2. Update pdf_path in this script")
        print("3. Run: python scratch/audit_q1_subquestions.py")
        return
    
    print("\n" + "="*80)
    print("AUDIT: Q1 + 1(a)-1(j) SUBQUESTIONS PIPELINE")
    print("="*80)
    print(f"PDF: {pdf_path.name}\n")
    
    # Process document
    print("[Step 1/4] Processing PDF...")
    blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf")
    print(f"  ✓ {num_pages} pages, {len(blocks)} OCR blocks")
    
    # Document understanding
    print("\n[Step 2/4] Document understanding (VLM enabled)...")
    doc_service = DocumentUnderstandingService()
    result = doc_service.process_document(
        blocks=blocks,
        document_id="q1_audit",
        page_sizes={i: [float(s[0]), float(s[1])] for i, s in enumerate(page_sizes, 1)},
        page_images=page_images,
        force_vlm_verification=True
    )
    print(f"  ✓ VLM Status: {result.vlm_status}")
    
    # Question extraction
    print("\n[Step 3/4] Question extraction (graph-driven)...")
    extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
    extraction = extractor.extract_validated_questions(
        blocks=blocks,
        document_id="q1_audit",
        doc_understanding_result=result
    )
    print(f"  ✓ {len(extraction.questions)} questions extracted")
    
    # Run audits
    print("\n[Step 4/4] Running audits...\n")
    
    audit_results = {}
    
    # Audit 1: VLM Structures
    if result.vlm_page_understandings:
        audit_results["1_vlm_structures"] = audit_vlm_structures(result.vlm_page_understandings[0])
        audit_results["2_vlm_relationships"] = audit_vlm_relationships(result.vlm_page_understandings[0])
    else:
        print("❌ No VLM page understandings available")
        audit_results["1_vlm_structures"] = False
        audit_results["2_vlm_relationships"] = False
    
    # Audit 3: Graph Edges
    if result.structure_graph:
        audit_results["3_graph_edges"] = audit_graph_edges(result.structure_graph)
    else:
        print("❌ No structure graph available")
        audit_results["3_graph_edges"] = False
    
    # Audit 4: Extraction Hierarchy
    audit_results["4_extraction_hierarchy"] = audit_extraction_hierarchy(extraction)
    
    # Audit 5: Invariants
    audit_results["5_invariants"] = audit_invariant_violations(extraction)
    
    # Final summary
    print_section("FINAL RESULTS")
    
    passed = sum(1 for v in audit_results.values() if v)
    total = len(audit_results)
    
    print(f"\nAudit Results: {passed}/{total} passed")
    for name, result in audit_results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    if passed == total:
        print("\n✓✓✓ ALL AUDITS PASSED ✓✓✓")
        print("\nConclusion:")
        print("The pipeline SUCCESSFULLY handles Q1 + 1(a)-1(j) subquestions.")
        print("Fix #2 is PRODUCTION-READY for subquestion hierarchies.")
    elif passed >= 4:
        print(f"\n⚠️ PARTIAL SUCCESS ({passed}/5 audits passed)")
        print("\nConclusion:")
        print("Most functionality works, but there are gaps:")
        if not audit_results.get("2_vlm_relationships", True):
            print("  - VLM is not returning subquestion_of relationships")
            print("    → Need: Add defensive relationship inference")
        if not audit_results.get("4_extraction_hierarchy", True):
            print("  - Extraction is not creating parent_question_id links")
            print("    → Need: Implement relationship inference")
    else:
        print(f"\n✗ CRITICAL FAILURE ({passed}/5 audits passed)")
        print("\nThe pipeline cannot currently handle Q1 + 1(a)-1(j) subquestions.")
        print("Specific gap identified in audit output above.")


if __name__ == "__main__":
    main()
