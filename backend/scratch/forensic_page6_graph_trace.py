"""
FORENSIC TRACE OF PAGE 6 GRAPH INTEGRITY BEFORE EXTRACTION.
Examines VLM output, Grounding, DocumentRegions, GraphNodes, and GraphEdges.
Traces provenance of all nodes and relationships.
"""
import sys
import os
import json
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.models.schemas import DocumentRegion, RegionRelationship, DocumentStructureGraph

pdf_path = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads/8a27cbab5a31/question_paper.pdf")
if not pdf_path.exists():
    print(f"ERROR: PDF not found at {pdf_path}")
    sys.exit(1)

print("="*80)
print(f"FORENSIC PAGE 6 GRAPH TRACE: {pdf_path.name}")
print("="*80)

# Process PDF
blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf", force_ocr=False)
p6_blocks = [b for b in blocks if b.page == 6]
p6_img = page_images.get(6) if page_images else None
p6_size = page_sizes[5] if len(page_sizes) >= 6 else (1000.0, 1000.0)

print(f"Total Pages: {num_pages}")
print(f"Page 6 OCR Blocks: {len(p6_blocks)}")
print(f"Page 6 Image: {'Present' if p6_img else 'None'} Size: {p6_size}")

# Run DocumentUnderstandingService on Page 6
doc_service = DocumentUnderstandingService()
page_sizes_dict = {6: [float(p6_size[0]), float(p6_size[1])]}
page_images_dict = {6: p6_img} if p6_img else {}

doc_result = doc_service.process_document(
    blocks=p6_blocks,
    document_id="p6_forensic_trace",
    page_sizes=page_sizes_dict,
    page_images=page_images_dict,
    force_vlm_verification=True,
)

graph: DocumentStructureGraph = doc_result.structure_graph
regions = doc_result.regions
relationships = doc_result.relationships

region_by_id = {r.region_id: r for r in regions}
node_by_id = graph.nodes

print("\n" + "="*80)
print("1. PAGE 6 GRAPH SUMMARY")
print("="*80)
print(f"Graph Semantic State: {graph.graph_semantic_state}")
print(f"Total Nodes on Page 6: {len(graph.nodes)}")
print(f"Total Edges on Page 6: {len(graph.edges)}")

# Roles count
roles_count = Counter(n.role for n in graph.nodes.values())
print("\nNodes by Role:")
for role, count in roles_count.most_common():
    print(f"  {role:20s}: {count}")

# Edges count by relationship type
edge_type_count = Counter(e.relationship for e in graph.edges)
print("\nEdges by Relationship Type:")
for rel, count in edge_type_count.most_common():
    print(f"  {rel:20s}: {count}")

# Adjacency maps
incoming = defaultdict(list)
outgoing = defaultdict(list)
for e in graph.edges:
    incoming[e.target_id].append(e)
    outgoing[e.source_id].append(e)

# Find Questions 1-8
q_nodes = sorted(
    [n for n in graph.nodes.values() if n.role == "QUESTION"],
    key=lambda x: (x.bbox.y, x.bbox.x)
)

print("\n" + "="*80)
print("2. QUESTIONS 1–8 DETAILED GRAPH NODES & EDGES AUDIT")
print("="*80)

q_node_ids = {qn.region_id for qn in q_nodes}

for i, qn in enumerate(q_nodes, start=1):
    q_id = qn.region_id
    q_reg = region_by_id.get(q_id)
    
    print(f"\n{'#'*40}")
    print(f"QUESTION #{i}: Node ID = {q_id}")
    print(f"{'#'*40}")
    print(f"  Role:              {qn.role}")
    print(f"  Page:              {qn.page}")
    print(f"  BBox:              (x={qn.bbox.x:.1f}, y={qn.bbox.y:.1f}, w={qn.bbox.width:.1f}, h={qn.bbox.height:.1f})")
    print(f"  Exact Text:        {qn.text!r}")
    print(f"  Semantic State:    {qn.semantic_state}")
    print(f"  Confidence:        {qn.confidence:.2f}")
    print(f"  Source Region IDs: {[q_id]}")
    
    # Check VLM provenance
    if q_reg:
        print(f"  Region Verification State: {q_reg.verification_state}")
        print(f"  Region Hypotheses:         {[h.hypothesized_type + ':' + h.source for h in q_reg.conflicting_hypotheses]}")
        print(f"  Region Evidence Count:     {len(q_reg.evidence)}")
        for ev in q_reg.evidence:
            print(f"    - Evidence: signal={ev.signal_type}, desc={ev.description!r}, score={ev.score}")
    
    # 1. OPTION nodes targeting this question
    inc_edges = incoming[q_id]
    out_edges = outgoing[q_id]
    
    option_edges = [e for e in inc_edges if e.relationship == "option_of"]
    subq_edges = [e for e in inc_edges if e.relationship == "subquestion_of"]
    cont_in_edges = [e for e in inc_edges if e.relationship == "continuation_of"]
    cont_out_edges = [e for e in out_edges if e.relationship == "continuation_of"]
    follows_in = [e for e in inc_edges if e.relationship == "follows"]
    follows_out = [e for e in out_edges if e.relationship == "follows"]
    
    print(f"\n  Connected OPTION Nodes (option_of -> {q_id}): {len(option_edges)} edges")
    for oe in option_edges:
        src_node = node_by_id.get(oe.source_id)
        role = src_node.role if src_node else "?"
        text = src_node.text if src_node else ""
        bbox = f"(x={src_node.bbox.x:.1f}, y={src_node.bbox.y:.1f}, w={src_node.bbox.width:.1f}, h={src_node.bbox.height:.1f})" if src_node else "?"
        sem_state = src_node.semantic_state if src_node else "?"
        conf = oe.confidence
        print(f"    * Node [{oe.source_id}] role={role:10s} conf={conf:.2f} sem_state={sem_state:10s} bbox={bbox}")
        print(f"      Text: {text!r}")
            
    print(f"\n  Connected SUBQUESTION Nodes (subquestion_of -> {q_id}): {len(subq_edges)} edges")
    for se in subq_edges:
        src_node = node_by_id.get(se.source_id)
        role = src_node.role if src_node else "?"
        text = src_node.text if src_node else ""
        bbox = f"(x={src_node.bbox.x:.1f}, y={src_node.bbox.y:.1f}, w={src_node.bbox.width:.1f}, h={src_node.bbox.height:.1f})" if src_node else "?"
        sem_state = src_node.semantic_state if src_node else "?"
        conf = se.confidence
        print(f"    * Node [{se.source_id}] role={role:12s} conf={conf:.2f} sem_state={sem_state:10s} bbox={bbox}")
        print(f"      Text: {text!r}")

    print(f"\n  Incoming CONTINUATION Edges (continuation_of -> {q_id}): {len(cont_in_edges)}")
    for ce in cont_in_edges:
        src_node = node_by_id.get(ce.source_id)
        role = src_node.role if src_node else "?"
        text = src_node.text if src_node else ""
        print(f"    * Node [{ce.source_id}] role={role} text={text!r}")

    print(f"  Outgoing CONTINUATION Edges ({q_id} -> continuation_of -> target): {len(cont_out_edges)}")
    for ce in cont_out_edges:
        tgt_node = node_by_id.get(ce.target_id)
        role = tgt_node.role if tgt_node else "?"
        text = tgt_node.text if tgt_node else ""
        print(f"    * Target [{ce.target_id}] role={role} text={text!r}")

    print(f"  Follows (Reading Order) In/Out: In={len(follows_in)}, Out={len(follows_out)}")

print("\n" + "="*80)
print("3. ALL RELATIONSHIPS INVOLVING Q1–Q8 (COMPLETE DUMP)")
print("="*80)

q_connected_nodes = set(q_node_ids)
for q_id in q_node_ids:
    for e in incoming[q_id] + outgoing[q_id]:
        q_connected_nodes.add(e.source_id)
        q_connected_nodes.add(e.target_id)

q_relationships = [
    e for e in graph.edges
    if e.source_id in q_connected_nodes or e.target_id in q_connected_nodes
]

print(f"Total relationships involving Q1–Q8 cluster: {len(q_relationships)}\n")
for e in q_relationships:
    src_node = node_by_id.get(e.source_id)
    tgt_node = node_by_id.get(e.target_id)
    src_role = src_node.role if src_node else "UNKNOWN"
    tgt_role = tgt_node.role if tgt_node else "UNKNOWN"
    src_text = (src_node.text[:25] if src_node else "")
    tgt_text = (tgt_node.text[:25] if tgt_node else "")
    print(f"  [{e.source_id}] ({src_role:10s} {src_text!r:27s}) --[{e.relationship:14s} conf={e.confidence:.2f}]--> [{e.target_id}] ({tgt_role:10s} {tgt_text!r:27s})")

print("\n" + "="*80)
print("4. DEEP ANOMALY INSPECTION")
print("="*80)

# Check 1: continuation_of from consumed OCR regions
print("\n[Check 1] Are there any continuation_of edges from constituent blocks inside grounded_text?")
consumed_cont_edges = []
for qn in q_nodes:
    for ce in incoming[qn.region_id]:
        if ce.relationship == "continuation_of":
            src = node_by_id.get(ce.source_id)
            if src and src.text in qn.text:
                consumed_cont_edges.append((qn, ce, src))
print(f"Result: Found {len(consumed_cont_edges)} consumed continuation edges.")
for qn, ce, src in consumed_cont_edges:
    print(f"  * QUESTION [{qn.region_id}] has continuation from [{ce.source_id}] text={src.text!r}")

# Check 2: continuation_of to options or another question
print("\n[Check 2] Are there continuation_of edges connecting to options or other questions?")
cross_role_cont = []
for e in graph.edges:
    if e.relationship == "continuation_of":
        src = node_by_id.get(e.source_id)
        tgt = node_by_id.get(e.target_id)
        if src and tgt:
            if src.role in ("OPTION", "QUESTION", "SUBQUESTION") or tgt.role in ("OPTION", "SUBQUESTION"):
                cross_role_cont.append((e, src, tgt))
print(f"Result: Found {len(cross_role_cont)} cross-role continuation edges on Page 6.")
for e, src, tgt in cross_role_cont:
    print(f"  * [{e.source_id}] ({src.role}) -> [{e.target_id}] ({tgt.role})")

# Check 3: Non-VLM grounded QUESTION nodes on Page 6
print("\n[Check 3] Are there any QUESTION nodes on Page 6 that were not created by VLM?")
non_vlm_q = []
for qn in q_nodes:
    reg = region_by_id.get(qn.region_id)
    if not reg or not any(ev.signal_type in ("visual_vlm_verification", "vlm_structural_authority") for ev in reg.evidence):
        non_vlm_q.append((qn, reg))
print(f"Result: Found {len(non_vlm_q)} non-VLM QUESTION nodes.")
for qn, reg in non_vlm_q:
    print(f"  * Node [{qn.region_id}] text={qn.text!r}")

# Check 4: UNKNOWN nodes with option_of or subquestion_of edges
print("\n[Check 4] Why do some UNKNOWN nodes have option_of or subquestion_of edges?")
unknown_opt_edges = [
    e for e in graph.edges
    if e.relationship in ("option_of", "subquestion_of")
    and node_by_id.get(e.source_id) and node_by_id[e.source_id].role == "UNKNOWN"
]
print(f"Result: Found {len(unknown_opt_edges)} option_of/subquestion_of edges originating from UNKNOWN nodes.")
for e in unknown_opt_edges[:10]:
    src = node_by_id[e.source_id]
    tgt = node_by_id.get(e.target_id)
    tgt_role = tgt.role if tgt else "?"
    print(f"  * Node [{e.source_id}] (UNKNOWN, text={src.text!r:15s}) --[{e.relationship}]--> [{e.target_id}] ({tgt_role})")

print("\n" + "="*80)
print("FORENSIC TRACE COMPLETE")
print("="*80)
