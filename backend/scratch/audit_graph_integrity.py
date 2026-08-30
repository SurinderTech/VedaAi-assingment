"""
READ-ONLY Graph Integrity Audit on Real Meghalaya PDF.
Strict audit of nodes, edges, relationships, and extraction consumption.
"""
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
from app.models.schemas import DocumentRegion, RegionRelationship, DocumentStructureGraph

pdf_path = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads/8a27cbab5a31/question_paper.pdf")
if not pdf_path.exists():
    print(f"ERROR: PDF not found at {pdf_path}")
    sys.exit(1)

print("="*80)
print(f"STARTING COMPREHENSIVE GRAPH AUDIT: {pdf_path.name}")
print("="*80)

# 1. Process Document
blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf", force_ocr=False)
print(f"Document processed: {len(blocks)} total OCR blocks across {num_pages} pages.")

# 2. Run Document Understanding Service on the FULL document
doc_service = DocumentUnderstandingService()
page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(page_sizes)}

doc_result = doc_service.process_document(
    blocks=blocks,
    document_id="audit_meghalaya",
    page_sizes=page_sizes_dict,
    page_images=page_images,
    force_vlm_verification=True
)

graph: DocumentStructureGraph = doc_result.structure_graph
print(f"\nDocument Understanding Complete.")
print(f"Graph Semantic State: {graph.graph_semantic_state}")
print(f"Total Graph Nodes:    {len(graph.nodes)}")
print(f"Total Graph Edges:    {len(graph.edges)}")

# ==============================================================================
# SECTION 1: Entire Document Relationship Types Breakdown
# ==============================================================================
print("\n" + "="*80)
print("SECTION 1: ENTIRE DOCUMENT RELATIONSHIP BREAKDOWN")
print("="*80)

all_edge_types = Counter(e.relationship for e in graph.edges)
for rel, count in all_edge_types.most_common():
    print(f"  {rel:25s}: {count:5d}")

# ==============================================================================
# SECTION 2: Page 6 Specific Nodes and Edges Breakdown
# ==============================================================================
print("\n" + "="*80)
print("SECTION 2: PAGE 6 SPECIFIC BREAKDOWN")
print("="*80)

p6_nodes = {nid: n for nid, n in graph.nodes.items() if n.page == 6}
p6_roles = Counter(n.role for n in p6_nodes.values())
print(f"Page 6 Total Nodes: {len(p6_nodes)}")
for role, count in p6_roles.most_common():
    print(f"  Role {role:20s}: {count:5d}")

p6_node_ids = set(p6_nodes.keys())
p6_edges = [
    e for e in graph.edges
    if e.source_id in p6_node_ids or e.target_id in p6_node_ids
]
p6_edge_types = Counter(e.relationship for e in p6_edges)
print(f"\nPage 6 Associated Edges: {len(p6_edges)}")
for rel, count in p6_edge_types.most_common():
    print(f"  {rel:25s}: {count:5d}")

# ==============================================================================
# SECTION 3: Per-Question (Q1-Q8 on Page 6) Detailed Edge Counts
# ==============================================================================
print("\n" + "="*80)
print("SECTION 3: QUESTIONS 1-8 ON PAGE 6 EDGE PROFILE")
print("="*80)

# Build fast adjacency indices
# target_id -> list of incoming edges (source, rel, conf)
incoming = defaultdict(list)
# source_id -> list of outgoing edges (target, rel, conf)
outgoing = defaultdict(list)

for e in graph.edges:
    incoming[e.target_id].append(e)
    outgoing[e.source_id].append(e)

# Find Q1-Q8 question nodes on Page 6
q_nodes_p6 = sorted(
    [n for n in p6_nodes.values() if n.role == "QUESTION"],
    key=lambda x: (x.bbox.y, x.bbox.x)
)

print(f"Found {len(q_nodes_p6)} QUESTION nodes on Page 6:\n")

for i, qn in enumerate(q_nodes_p6, start=1):
    q_id = qn.region_id
    inc = incoming[q_id]
    outg = outgoing[q_id]

    inc_rel_counts = Counter(e.relationship for e in inc)
    outg_rel_counts = Counter(e.relationship for e in outg)

    # option_of targeting this question
    n_option_of = inc_rel_counts.get("option_of", 0)
    # subquestion_of targeting this question
    n_subquestion_of = inc_rel_counts.get("subquestion_of", 0)
    # continuation_of targeting this question
    n_cont_in = inc_rel_counts.get("continuation_of", 0)
    n_cont_out = outg_rel_counts.get("continuation_of", 0)
    # follows in/out
    n_follows_in = inc_rel_counts.get("follows", 0)
    n_follows_out = outg_rel_counts.get("follows", 0)
    # contains in/out
    n_contains_in = inc_rel_counts.get("contains", 0)
    n_contains_out = outg_rel_counts.get("contains", 0)

    print(f"Question #{i} (Node ID: {q_id}, BBox: y={qn.bbox.y:4.0f}, text={qn.text[:45]!r}):")
    print(f"  - option_of (incoming)       : {n_option_of}")
    print(f"  - subquestion_of (incoming)  : {n_subquestion_of}")
    print(f"  - continuation_of (incoming) : {n_cont_in}")
    print(f"  - continuation_of (outgoing) : {n_cont_out}")
    print(f"  - follows (incoming)         : {n_follows_in}")
    print(f"  - follows (outgoing)         : {n_follows_out}")
    print(f"  - contains (incoming)        : {n_contains_in}")
    print(f"  - contains (outgoing)        : {n_contains_out}")

    # Detailed dump of option and subquestion nodes
    if n_option_of > 0:
        opt_sources = [e.source_id for e in inc if e.relationship == "option_of"]
        print(f"    Options attached ({len(opt_sources)}):")
        for osid in opt_sources:
            onode = graph.nodes.get(osid)
            otext = onode.text[:40] if onode else "UNKNOWN_NODE"
            print(f"      * [{osid}] role={onode.role if onode else '?'} text={otext!r}")

    if n_subquestion_of > 0:
        sq_sources = [e.source_id for e in inc if e.relationship == "subquestion_of"]
        print(f"    Subquestions attached ({len(sq_sources)}):")
        for sqid in sq_sources:
            sqnode = graph.nodes.get(sqid)
            sqtext = sqnode.text[:40] if sqnode else "UNKNOWN_NODE"
            print(f"      * [{sqid}] role={sqnode.role if sqnode else '?'} text={sqtext!r}")

    print()

# ==============================================================================
# SECTION 4: Structural Anomaly Check
# ==============================================================================
print("="*80)
print("SECTION 4: STRUCTURAL ANOMALY DETECTION (Document-Wide & Page 6)")
print("="*80)

# Check A: Suspicious continuation_of edges
suspicious_continuations = []
for e in graph.edges:
    if e.relationship == "continuation_of":
        src = graph.nodes.get(e.source_id)
        tgt = graph.nodes.get(e.target_id)
        # Suspicious if src is QUESTION or SECTION_HEADER or same ID
        if e.source_id == e.target_id:
            suspicious_continuations.append((e, "Self-loop continuation"))
        elif src and src.role in ("QUESTION", "SECTION_HEADER") and tgt and tgt.role == "QUESTION":
            suspicious_continuations.append((e, f"Suspicious continuation between {src.role} and {tgt.role}"))
print(f"Suspicious continuation_of edges found: {len(suspicious_continuations)}")
for sc, reason in suspicious_continuations:
    print(f"  * {sc.source_id} -> {sc.target_id}: {reason}")

# Check B: Options with no parent (or multiple parents)
all_option_nodes = [n for n in graph.nodes.values() if n.role == "OPTION"]
options_without_parent = []
options_with_multi_parent = []
for onode in all_option_nodes:
    parents = [e.target_id for e in outgoing[onode.region_id] if e.relationship == "option_of"]
    if not parents:
        options_without_parent.append(onode)
    elif len(parents) > 1:
        options_with_multi_parent.append((onode, parents))

print(f"\nOptions without parent QUESTION (total {len(all_option_nodes)} OPTION nodes): {len(options_without_parent)}")
for onode in options_without_parent[:10]:
    print(f"  * Page {onode.page} [{onode.region_id}] text={onode.text[:40]!r}")

print(f"Options with MULTIPLE parent QUESTIONs: {len(options_with_multi_parent)}")
for onode, parents in options_with_multi_parent:
    print(f"  * Page {onode.page} [{onode.region_id}] parents={parents} text={onode.text[:40]!r}")

# Check C: Subquestions with no parent (or multiple parents)
all_subq_nodes = [n for n in graph.nodes.values() if n.role == "SUBQUESTION"]
subq_without_parent = []
subq_with_multi_parent = []
for sqnode in all_subq_nodes:
    parents = [e.target_id for e in outgoing[sqnode.region_id] if e.relationship == "subquestion_of"]
    if not parents:
        subq_without_parent.append(sqnode)
    elif len(parents) > 1:
        subq_with_multi_parent.append((sqnode, parents))

print(f"\nSubquestions without parent QUESTION (total {len(all_subq_nodes)} SUBQUESTION nodes): {len(subq_without_parent)}")
for sqnode in subq_without_parent[:10]:
    print(f"  * Page {sqnode.page} [{sqnode.region_id}] text={sqnode.text[:40]!r}")

print(f"Subquestions with MULTIPLE parent QUESTIONs: {len(subq_with_multi_parent)}")
for sqnode, parents in subq_with_multi_parent:
    print(f"  * Page {sqnode.page} [{sqnode.region_id}] parents={parents} text={sqnode.text[:40]!r}")

# Check D: Self-loops and duplicate edges
self_loops = [e for e in graph.edges if e.source_id == e.target_id]
seen_edges = set()
duplicate_edges = []
for e in graph.edges:
    key = (e.source_id, e.target_id, e.relationship)
    if key in seen_edges:
        duplicate_edges.append(e)
    seen_edges.add(key)

print(f"\nSelf-loops: {len(self_loops)}")
print(f"Duplicate edges: {len(duplicate_edges)}")

# ==============================================================================
# SECTION 5: Verify Extraction Consumption of follows and UNKNOWN Nodes
# ==============================================================================
print("\n" + "="*80)
print("SECTION 5: EXTRACTION CONSUMPTION VERIFICATION")
print("="*80)

extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
extraction_result = extractor.extract_validated_questions(
    blocks=blocks,
    document_id="audit_meghalaya",
    doc_understanding_result=doc_result,
)

print(f"Total Extracted Questions Document-Wide: {len(extraction_result.questions)}")
p6_extracted = [q for q in extraction_result.questions if q.page == 6]
print(f"Total Extracted Questions on Page 6:     {len(p6_extracted)}")

# Check if any UNKNOWN OCR region was promoted to a QUESTION
unknown_as_question = []
for q in extraction_result.questions:
    primary_id = q.source_region_ids[0] if q.source_region_ids else None
    node = graph.nodes.get(primary_id)
    if node and node.role == "UNKNOWN":
        unknown_as_question.append((q, node))

print(f"\nUNKNOWN regions emitted as QUESTIONs: {len(unknown_as_question)}")
for q, node in unknown_as_question:
    print(f"  * Q-ID: {q.id}, Page: {q.page}, Text: {q.text[:50]!r}")

# ==============================================================================
# SECTION 6: Complete Pipeline Question Summary
# ==============================================================================
print("\n" + "="*80)
print("SECTION 6: COMPLETE DOCUMENT QUESTION COUNTS BY PAGE")
print("="*80)

page_q_counts = Counter(q.page for q in extraction_result.questions)
for p in range(1, num_pages + 1):
    qs = [q for q in extraction_result.questions if q.page == p]
    mcqs = sum(1 for q in qs if q.question_type == "MCQ")
    short_qs = sum(1 for q in qs if q.question_type == "SHORT_ANSWER")
    long_qs = sum(1 for q in qs if q.question_type == "LONG_ANSWER")
    print(f"  Page {p:2d}: {len(qs):2d} questions (MCQ={mcqs:2d}, Short={short_qs:2d}, Long={long_qs:2d})")

print("\n" + "="*80)
print("AUDIT SCRIPT EXECUTION COMPLETE")
print("="*80)
