"""
Real Document Graph & Extraction Audit for Fix #2.
Processes the Meghalaya 12-page PDF and audits:
1. QUESTION, OPTION, SUBQUESTION graph node counts
2. continuation_of, option_of, subquestion_of edge counts
3. Graph semantic state
4. Page-level completeness
5. Check for any standalone garbage question fragments ("What", "which", "List", "Why")
6. Final extracted question count and list
"""
import sys
import os
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["DOCUMENT_VLM_ENABLED"] = "false"
os.environ["DOCUMENT_VLM_PAGE_UNDERSTANDING"] = "false"

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService

pdf_path = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads/8a27cbab5a31/question_paper.pdf")
if not pdf_path.exists():
    pdf_path = Path(__file__).parent / "test_corpus" / "multi_page_paper.pdf"

print("="*80)
print(f"REAL DOCUMENT VALIDATION: {pdf_path.name}")
print(f"Path: {pdf_path} ({pdf_path.stat().st_size} bytes)")
print("="*80)

blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf", force_ocr=False)
print(f"\n[1] Extracted {len(blocks)} OCR blocks across {num_pages} pages.")

doc_service = DocumentUnderstandingService()
result = doc_service.process_document(
    blocks=blocks,
    document_id="meghalaya_audit",
    page_sizes={i: [float(s[0]), float(s[1])] for i, s in enumerate(page_sizes, 1)},
    page_images=page_images,
    force_vlm_verification=False  # Test normal/deterministic fallback path
)

graph = result.structure_graph
print(f"\n[2] DocumentStructureGraph built:")
print(f"    Graph Semantic State: {graph.graph_semantic_state}")
print(f"    Total Nodes: {len(graph.nodes)}")
print(f"    Total Edges: {len(graph.edges)}")

# Count node roles
role_counts = {}
for node in graph.nodes.values():
    role_counts[node.role] = role_counts.get(node.role, 0) + 1

print(f"\n[3] Node Role Breakdown:")
for role, count in sorted(role_counts.items()):
    print(f"    {role:20s}: {count}")

# Count edge relationships
edge_counts = {}
for edge in graph.edges:
    edge_counts[edge.relationship] = edge_counts.get(edge.relationship, 0) + 1

print(f"\n[4] Edge Relationship Breakdown:")
for rel, count in sorted(edge_counts.items()):
    print(f"    {rel:20s}: {count}")

# Audit for garbage / fragment questions
garbage_words = ["what", "which", "list", "why", "write", "explain", "describe", "define"]
garbage_nodes = []
for node in graph.nodes.values():
    if node.role == "QUESTION":
        t = node.text.strip().lower()
        if t in garbage_words:
            garbage_nodes.append((node.region_id, node.text, node.page))

print(f"\n[5] Garbage / Fragment QUESTION Node Check:")
if garbage_nodes:
    print(f"    WARNING: Found {len(garbage_nodes)} fragment QUESTION nodes:")
    for gid, gt, gp in garbage_nodes:
        print(f"      - [{gid}] (p.{gp}): {gt!r}")
else:
    print(f"    SUCCESS: 0 fragment QUESTION nodes found! No standalone 'What', 'which', 'List', 'Why' nodes.")

# Question Extraction
extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
extraction = extractor.extract_validated_questions(
    blocks=blocks,
    document_id="meghalaya_audit",
    doc_understanding_result=result,
)

print(f"\n[6] Question Extraction Results:")
print(f"    Total Top-Level Questions: {len(extraction.questions)}")
print(f"    Audit Accepted Count:      {extraction.audit.accepted_question_count}")
print(f"    Audit Rejected Count:      {extraction.audit.rejected_count}")
print(f"    Audit Duplicate Rejected:  {extraction.audit.duplicate_rejected}")
print(f"    Audit Invariant Violations: {len(extraction.audit.invariant_violations)}")

print(f"\n[7] Sample Extracted Questions:")
for q in extraction.questions[:15]:
    opt_info = f"({len(q.options)} options)" if q.options else ""
    sub_info = f"({len(q.subquestions)} subquestions)" if getattr(q, 'subquestions', None) else ""
    print(f"    Q{q.number:6s} [{q.question_type:12s}] {opt_info} {sub_info} p.{q.page}: {q.text[:65]!r}")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
