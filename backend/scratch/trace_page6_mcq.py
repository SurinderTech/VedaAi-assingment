"""
Trace Page 6 MCQ Questions 1-8 through every stage of the document intelligence pipeline.
Stages:
1. VLM Response for Page 6
2. _ground_structure_to_ocr
3. _apply_vlm_page_understandings
4. _build_structure_graph
5. _extract_from_graph

Dumps all nodes, region IDs, bboxes, texts, confidence, semantic state, and edge types for Q1-Q8.
"""
import sys
import os
import io
import json
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env
import app.core.config
from app.core.config import settings
from app.services.document_processor import process_document
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService

pdf_path = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads/8a27cbab5a31/question_paper.pdf")
if not pdf_path.exists():
    pdf_path = Path(__file__).parent / "test_corpus" / "multi_page_paper.pdf"

print("="*80)
print(f"PAGE 6 DETAILED TRACE ON REAL DOCUMENT: {pdf_path.name}")
print("="*80)

# Process Page 6
blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf", force_ocr=False)
p6_blocks = [b for b in blocks if b.page == 6]
p6_img = page_images.get(6) if page_images else None
p6_size = page_sizes[5] if len(page_sizes) >= 6 else (1000.0, 1000.0)

print(f"\n[STAGE 1] Page 6 OCR Blocks: {len(p6_blocks)} blocks, image={'Present' if p6_img else 'None'}")
for i, b in enumerate(p6_blocks[:15]):
    print(f"  OCR[{i:02d}] id={b.id:8s} bbox=({b.bbox.x:4.0f},{b.bbox.y:4.0f},{b.bbox.width:4.0f},{b.bbox.height:4.0f}) conf={b.confidence:.2f} text={b.text[:50]!r}")

# Stage 2: Run VLM on Page 6
print("\n" + "="*80)
print("[STAGE 2] VLM PAGE UNDERSTANDING ON PAGE 6:")
print("="*80)

vision_provider = MultimodalDocumentVisionProvider()
p6_understanding = vision_provider.understand_page(
    page_image=p6_img,
    ocr_blocks=p6_blocks,
    page_number=6,
    total_pages=num_pages,
    page_context={"prev_page_last_type": "SECTION_HEADER", "next_page_first_type": "QUESTION"},
    force_vlm=True
)

print(f"  VLM Result:             {p6_understanding.vlm_result}")
print(f"  VLM Provider / Model:   {p6_understanding.vlm_provider} / {p6_understanding.vlm_model}")
print(f"  Finish Reason:          {p6_understanding.finish_reason}")
print(f"  Semantic Completeness:  {p6_understanding.semantic_completeness}")
print(f"  Structures Produced:    {len(p6_understanding.structures)}")
print(f"  Relationships Produced: {len(p6_understanding.relationships)}")

print("\n  VLM STRUCTURES DUMP (First 20):")
for i, s in enumerate(p6_understanding.structures[:20]):
    print(f"    [{i:02d}] role={s.role:15s} num={s.display_number or '-':4s} lbl={s.display_label or '-':4s} status={s.grounding_status:12s} grounded_ids={s.grounded_region_ids} text={s.grounded_text[:40]!r}")

print("\n  VLM RELATIONSHIPS DUMP:")
for i, r in enumerate(p6_understanding.relationships[:20]):
    print(f"    [{i:02d}] {r.source_ids} --[{r.relationship_type}]--> {r.target_ids}")

# Stage 3: Document Understanding & Grounding
print("\n" + "="*80)
print("[STAGE 3] _ground_structure_to_ocr & _apply_vlm_page_understandings:")
print("="*80)

doc_service = DocumentUnderstandingService()
page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(page_sizes)}

# We process just page 6 blocks to isolate page 6
doc_result = doc_service.process_document(
    blocks=p6_blocks,
    document_id="page6_trace",
    page_sizes={6: [float(p6_size[0]), float(p6_size[1])]},
    page_images={6: p6_img} if p6_img else {},
    force_vlm_verification=True
)

print(f"  Total Regions: {len(doc_result.regions)}")
print(f"  Total Relationships: {len(doc_result.relationships)}")

# Stage 4: Structure Graph Nodes & Edges for Page 6
print("\n" + "="*80)
print("[STAGE 4] _build_structure_graph DUMP (Page 6 Nodes & Edges):")
print("="*80)

graph = doc_result.structure_graph
print(f"  Graph Semantic State: {graph.graph_semantic_state}")
print(f"  Total Graph Nodes:    {len(graph.nodes)}")
print(f"  Total Graph Edges:    {len(graph.edges)}")

print("\n  ALL QUESTION, OPTION, SUBQUESTION NODES ON PAGE 6:")
semantic_nodes = [n for n in graph.nodes.values() if n.role in ("QUESTION", "OPTION", "SUBQUESTION")]
for n in sorted(semantic_nodes, key=lambda x: (x.bbox.y if x.bbox else 0, x.bbox.x if x.bbox else 0)):
    print(f"    Node [{n.region_id:8s}] role={n.role:12s} p.{n.page} conf={n.confidence:.2f} state={n.semantic_state:10s} bbox=({n.bbox.x:4.0f},{n.bbox.y:4.0f},{n.bbox.width:4.0f},{n.bbox.height:4.0f}) text={n.text[:60]!r}")

print("\n  ALL EDGES CONNECTING TO QUESTION/OPTION/SUBQUESTION NODES:")
for e in graph.edges:
    src_node = graph.nodes.get(e.source_id)
    tgt_node = graph.nodes.get(e.target_id)
    if (src_node and src_node.role in ("QUESTION", "OPTION", "SUBQUESTION")) or (tgt_node and tgt_node.role in ("QUESTION", "OPTION", "SUBQUESTION")):
        src_role = src_node.role if src_node else "?"
        tgt_role = tgt_node.role if tgt_node else "?"
        print(f"    Edge: [{e.source_id:8s}] ({src_role}) --[{e.relationship:15s} conf={e.confidence:.2f} state={e.semantic_state}]--> [{e.target_id:8s}] ({tgt_role})")

# Stage 5: Extraction from Graph
print("\n" + "="*80)
print("[STAGE 5] _extract_from_graph EXTRACTION RESULTS:")
print("="*80)

extractor = IntelligentQuestionExtractionService(doc_understanding_service=doc_service)
extraction = extractor.extract_validated_questions(
    blocks=p6_blocks,
    document_id="page6_trace",
    doc_understanding_result=doc_result,
    page_sizes={6: [float(p6_size[0]), float(p6_size[1])]},
)

print(f"  Total Extracted Questions on Page 6: {len(extraction.questions)}")
for q in extraction.questions:
    print(f"\n  Extracted Question Q{q.number}:")
    print(f"    ID:       {q.id}")
    print(f"    Type:     {q.question_type}")
    print(f"    Page:     {q.page}")
    print(f"    Text:     {q.text!r}")
    print(f"    Options ({len(q.options)}):")
    for opt in q.options:
        print(f"      - {opt}")
    if getattr(q, 'subquestions', None):
        print(f"    Subquestions ({len(q.subquestions)}):")
        for sq in q.subquestions:
            print(f"      * Q{sq.number}: {sq.text[:50]!r}")

print("\n" + "="*80)
print("TRACE COMPLETE")
print("="*80)
