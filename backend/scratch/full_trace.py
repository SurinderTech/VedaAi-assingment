"""
FULL END-TO-END PIPELINE TRACE
Runs the real uploaded question paper PDF through every stage and captures
concrete evidence at each boundary. No code is modified.

Usage:
    py scratch\full_trace.py <assessment_id>

This script re-runs ONLY the document-understanding + extraction stages
(Steps 1-3 of the pipeline) using the already-uploaded files from the store,
and captures every intermediate state.
"""
from __future__ import annotations
import sys, os, json

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def _s(t, n=80):
    """Safe truncate with ascii-safe repr."""
    return repr(str(t)[:n]).replace('\\x', '?')

sys.path.insert(0, ".")
os.environ.setdefault("PRIMARY_LLM_PROVIDER", "gemini")
os.environ.setdefault("DOCUMENT_VLM_ENABLED", "true")
os.environ.setdefault("DOCUMENT_VLM_PAGE_UNDERSTANDING", "true")

# Load real API keys from .env manually
import pathlib
env_file = pathlib.Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

assessment_id = sys.argv[1] if len(sys.argv) > 1 else None
if not assessment_id:
    print("Usage: py scratch/full_trace.py <assessment_id>")
    sys.exit(1)

from app.core import store
files = store.get_files(assessment_id)
if not files:
    print(f"No files found for assessment_id={assessment_id}")
    sys.exit(1)

qp_path = files["question_paper"]
qp_ext  = files["question_paper_ext"]
print(f"\n[INPUT] Question paper: {qp_path}  ext={qp_ext}")
print(f"        File exists: {os.path.exists(qp_path)}")

# ─────────────────────────────────────────────────────────────
# STAGE 1: PDF rendering + OCR / native text extraction
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 1: PDF RENDERING + OCR/NATIVE TEXT EXTRACTION")
print(f"{'='*60}")

from app.services.document_processor import process_document
blocks, num_pages, page_sizes, page_images_dict = process_document(qp_path, qp_ext, False)

print(f"  Pages rendered: {num_pages}")
print(f"  Page images captured: {len(page_images_dict)}")
print(f"  Total OCR blocks: {len(blocks)}")

pages_dict = {}
for b in blocks:
    pages_dict.setdefault(b.page, []).append(b)

for pg in sorted(pages_dict.keys()):
    pg_blocks = pages_dict[pg]
    img = page_images_dict.get(pg)
    img_sz = img.size if img else "NO IMAGE"
    print(f"  Page {pg:02d}: {len(pg_blocks):3d} OCR blocks | image={img_sz}")
    # Show first 5 blocks to see what OCR produced
    for b in pg_blocks[:5]:
        try:
            safe_text = b.text[:60].encode('ascii', errors='replace').decode('ascii')
        except Exception:
            safe_text = "<encode error>"
        print(f"           [{b.id}] src={b.source:10s} conf={b.confidence:.2f} "
              f"bbox=({b.bbox.x:.0f},{b.bbox.y:.0f},{b.bbox.width:.0f},{b.bbox.height:.0f}) "
              f"text={safe_text!r}")
    if len(pg_blocks) > 5:
        print(f"           ... and {len(pg_blocks)-5} more blocks")

# ─────────────────────────────────────────────────────────────
# STAGE 2: DocumentUnderstanding — instrument with hooks
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 2: DOCUMENT UNDERSTANDING (VLM + DETERMINISTIC)")
print(f"{'='*60}")

import app.services.document_understanding_service as dus_mod
import app.services.document_vision_provider as dvp_mod

# --- Monkey-patch _parse_page_understanding to capture raw VLM responses ---
_captured_vlm = {}   # page_num → {raw_response, structures, relationships, finish_reason}
_original_parse = dvp_mod.MultimodalDocumentVisionProvider._parse_page_understanding

def _patched_parse(self, response_text, page_number, ocr_blocks, page_b64_sent, vlm_meta=None):
    result = _original_parse(self, response_text, page_number, ocr_blocks, page_b64_sent, vlm_meta)
    _captured_vlm[page_number] = {
        "raw_response_len": len(response_text),
        "raw_response_first_500": response_text[:500],
        "raw_response_last_300": response_text[-300:] if len(response_text) > 300 else "",
        "finish_reason": result.finish_reason,
        "vlm_result": result.vlm_result,
        "structures_produced": result.structures_produced,
        "relationships_produced": result.relationships_produced,
        "structures": [
            {
                "region_ids": s.region_ids,
                "role": s.role,
                "display_number": s.display_number,
                "display_label": s.display_label,
                "confidence": s.confidence,
                "reasoning": s.reasoning[:80],
                "grounded_region_ids": s.grounded_region_ids,
                "grounding_status": s.grounding_status,
            }
            for s in result.structures
        ],
        "relationships": [
            {
                "source_ids": r.source_ids,
                "target_ids": r.target_ids,
                "type": r.relationship_type,
                "confidence": r.confidence,
            }
            for r in result.relationships
        ],
    }
    return result

dvp_mod.MultimodalDocumentVisionProvider._parse_page_understanding = _patched_parse

# --- Monkey-patch _apply_vlm_page_understandings to capture before/after region states ---
_captured_apply = {}  # page_num → {before, after, vlm_named_ids}
_original_apply = dus_mod.DocumentUnderstandingService._apply_vlm_page_understandings

def _patched_apply(self, vlm_understandings, all_regions, all_relationships, *args, **kwargs):
    # Capture BEFORE state per page
    before_by_page = {}
    for pg_num in set(r.page for r in all_regions):
        before_by_page[pg_num] = {
            r.region_id: {"type": r.region_type, "text": r.text[:60], "conf": r.confidence}
            for r in all_regions if r.page == pg_num
        }

    pages_dict_arg = kwargs.get("pages_dict", args[0] if args else {})
    _original_apply(self, vlm_understandings, all_regions, all_relationships, pages_dict_arg)

    # Capture AFTER state per page
    after_by_page = {}
    for pg_num in set(r.page for r in all_regions):
        after_by_page[pg_num] = {
            r.region_id: {"type": r.region_type, "text": r.text[:60], "conf": r.confidence,
                          "verification_state": r.verification_state}
            for r in all_regions if r.page == pg_num
        }

    for pg_num in set(list(before_by_page.keys()) + list(after_by_page.keys())):
        before = before_by_page.get(pg_num, {})
        after  = after_by_page.get(pg_num, {})
        changed, unchanged_q = {}, []
        for rid in set(list(before.keys()) + list(after.keys())):
            b = before.get(rid, {})
            a = after.get(rid, {})
            if b.get("type") != a.get("type"):
                changed[rid] = {"before": b.get("type"), "after": a.get("type"), "text": b.get("text", a.get("text",""))}
            elif a.get("type") == "QUESTION":
                unchanged_q.append({"rid": rid, "text": a.get("text",""), "type": "QUESTION"})
        _captured_apply[pg_num] = {
            "region_count_before": len(before),
            "region_count_after": len(after),
            "role_changes": changed,
            "unchanged_QUESTION_regions": unchanged_q,
        }

dus_mod.DocumentUnderstandingService._apply_vlm_page_understandings = _patched_apply

# --- Monkey-patch _build_structure_graph to capture graph contents ---
_captured_graph = {}
_original_build = dus_mod.DocumentUnderstandingService._build_structure_graph

def _patched_build(self, all_regions, all_relationships, document_purpose, page_roles):
    graph = _original_build(self, all_regions, all_relationships, document_purpose, page_roles)
    _captured_graph["nodes_total"] = len(graph.nodes)
    _captured_graph["edges_total"] = len(graph.edges)
    _captured_graph["graph_state"] = graph.graph_semantic_state
    _captured_graph["document_purpose"] = graph.document_purpose
    # Classify nodes by role
    role_counts = {}
    for n in graph.nodes.values():
        role_counts[n.role] = role_counts.get(n.role, 0) + 1
    _captured_graph["role_counts"] = role_counts
    # Classify edges by type and evidence source
    edge_type_counts = {}
    for e in graph.edges:
        edge_type_counts[e.relationship] = edge_type_counts.get(e.relationship, 0) + 1
    _captured_graph["edge_type_counts"] = edge_type_counts
    # Capture per-page node summary
    nodes_by_page = {}
    for n in graph.nodes.values():
        nodes_by_page.setdefault(n.page, []).append({
            "rid": n.region_id, "role": n.role, "state": n.semantic_state,
            "conf": round(n.confidence, 2), "text": n.text[:60]
        })
    _captured_graph["nodes_by_page"] = nodes_by_page
    # Capture edge sources
    vlm_edges = [e for e in graph.edges if any("visual_vlm_verification" in s for s in e.evidence_sources)]
    det_edges  = [e for e in graph.edges if not any("visual_vlm_verification" in s for s in e.evidence_sources)]
    _captured_graph["vlm_edge_count"] = len(vlm_edges)
    _captured_graph["deterministic_edge_count"] = len(det_edges)
    return graph

dus_mod.DocumentUnderstandingService._build_structure_graph = _patched_build

# Now run the actual document understanding
page_sizes_dict = {i+1: [float(w), float(h)] for i, (w, h) in enumerate(page_sizes)} if page_sizes else None

from app.services.document_understanding_service import DocumentUnderstandingService
du_svc = DocumentUnderstandingService()
doc_understanding_result = du_svc.process_document(
    blocks=blocks,
    document_id=f"trace_{assessment_id}",
    page_sizes=page_sizes_dict,
    page_images=page_images_dict,
)

# ─────────────────────────────────────────────────────────────
# STAGE 2 REPORT: VLM responses per page
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("STAGE 2A: VLM RAW RESPONSE PER PAGE")
print(f"{'─'*60}")

for pg in sorted(_captured_vlm.keys()):
    v = _captured_vlm[pg]
    print(f"\n  PAGE {pg}: finish_reason={v['finish_reason']} | vlm_result={v['vlm_result']} | "
          f"structures={v['structures_produced']} | relationships={v['relationships_produced']} | "
          f"raw_len={v['raw_response_len']}")
    if v["structures_produced"] == 0:
        print(f"    ⚠️  NO STRUCTURES RETURNED")
        print(f"    raw_first_500: {v['raw_response_first_500']!r}")
    else:
        for s in v["structures"]:
            print(f"    struct: role={s['role']:15s} conf={s['confidence']:.2f} "
                  f"ids={s['region_ids']} display={s['display_number']} "
                  f"reasoning={s['reasoning']!r}")
        for r in v["relationships"]:
            print(f"    rel:   {r['source_ids']} --{r['type']}--> {r['target_ids']} conf={r['confidence']:.2f}")
    if v["finish_reason"] == "MAX_TOKENS":
        print(f"    ⚠️  MAX_TOKENS: response was TRUNCATED")
        print(f"    last_300: {v['raw_response_last_300']!r}")

# ─────────────────────────────────────────────────────────────
# STAGE 2B REPORT: Region state changes per page (before/after VLM)
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("STAGE 2B: REGION STATE CHANGES (before/after VLM application)")
print(f"{'─'*60}")

for pg in sorted(_captured_apply.keys()):
    a = _captured_apply[pg]
    changed = a["role_changes"]
    ghost_q  = a["unchanged_QUESTION_regions"]
    print(f"\n  PAGE {pg}: regions_before={a['region_count_before']} "
          f"regions_after={a['region_count_after']} | "
          f"role_changes={len(changed)} | "
          f"unchanged_QUESTION_ghost_nodes={len(ghost_q)}")
    if changed:
        for rid, ch in list(changed.items())[:10]:
            print(f"    CHANGED {ch['before']:15s} → {ch['after']:15s}  [{rid}] {ch['text']!r}")
        if len(changed) > 10:
            print(f"    ... and {len(changed)-10} more changes")
    if ghost_q:
        print(f"    GHOST QUESTION nodes (not addressed by VLM, kept QUESTION):")
        for g in ghost_q[:15]:
            print(f"      [{g['rid']}] {g['text']!r}")
        if len(ghost_q) > 15:
            print(f"      ... and {len(ghost_q)-15} more ghost QUESTIONs")

# ─────────────────────────────────────────────────────────────
# STAGE 2C REPORT: DocumentStructureGraph
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("STAGE 2C: DOCUMENTSTRUCTUREGRAPH")
print(f"{'─'*60}")
g = _captured_graph
print(f"  graph_state={g.get('graph_state')} | document_purpose={g.get('document_purpose')}")
print(f"  Total nodes: {g.get('nodes_total')}  Total edges: {g.get('edges_total')}")
print(f"  VLM edges:   {g.get('vlm_edge_count')}  Deterministic edges: {g.get('deterministic_edge_count')}")
print(f"  Role counts: {g.get('role_counts')}")
print(f"  Edge types:  {g.get('edge_type_counts')}")

print(f"\n  Nodes by page:")
nodes_by_page = g.get("nodes_by_page", {})
for pg in sorted(nodes_by_page.keys()):
    page_nodes = nodes_by_page[pg]
    q_nodes = [n for n in page_nodes if n["role"] == "QUESTION"]
    other   = [n for n in page_nodes if n["role"] != "QUESTION"]
    print(f"    Page {pg:02d}: {len(page_nodes):3d} nodes | QUESTION={len(q_nodes)} | other_roles={len(other)}")
    for n in q_nodes[:8]:
        print(f"      Q-node [{n['rid']}] conf={n['conf']} state={n['state']} text={n['text']!r}")
    if len(q_nodes) > 8:
        print(f"      ... and {len(q_nodes)-8} more QUESTION nodes")

# ─────────────────────────────────────────────────────────────
# STAGE 3: IntelligentQuestionExtraction
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 3: INTELLIGENT QUESTION EXTRACTION")
print(f"{'='*60}")

from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
iq_svc = IntelligentQuestionExtractionService(doc_understanding_service=du_svc)

# Patch _extract_from_graph to capture which path was taken
_captured_extraction = {}
_orig_extract_graph = iq_svc._extract_from_graph.__func__

def _patched_extract_graph(self, graph, doc_result, document_id):
    q_nodes = [n for n in graph.nodes.values() if n.role == "QUESTION"]
    _captured_extraction["graph_q_nodes"] = len(q_nodes)
    _captured_extraction["graph_state"] = graph.graph_semantic_state
    result = _orig_extract_graph(self, graph, doc_result, document_id)
    _captured_extraction["extraction_path"] = "graph"
    _captured_extraction["extracted_count"] = len(result.questions)
    _captured_extraction["fallback_used"] = result.fallback_used
    _captured_extraction["invariant_violations"] = result.invariant_violations
    return result

iq_svc._extract_from_graph = lambda graph, doc_result, document_id: \
    _patched_extract_graph(iq_svc, graph, doc_result, document_id)

import asyncio
questions = asyncio.run(
    __import__("app.services.question_extractor", fromlist=["extract_questions"]).extract_questions(
        blocks, doc_understanding_result=doc_understanding_result, page_sizes=page_sizes_dict
    )
)

print(f"  Extraction path used: {_captured_extraction.get('extraction_path', 'unknown')}")
print(f"  Graph QUESTION nodes seen by extraction: {_captured_extraction.get('graph_q_nodes', '?')}")
print(f"  Graph state at extraction: {_captured_extraction.get('graph_state', '?')}")
print(f"  Fallback used: {_captured_extraction.get('fallback_used', '?')}")
print(f"  Questions extracted: {_captured_extraction.get('extracted_count', len(questions))}")
if _captured_extraction.get("invariant_violations"):
    print(f"  ⚠️  Invariant violations: {_captured_extraction['invariant_violations']}")

print(f"\n  All {len(questions)} extracted questions:")
for i, q in enumerate(questions):
    opts = q.options if hasattr(q, "options") else []
    print(f"  [{i+1:02d}] Q{q.number} pg={q.page} type={getattr(q,'question_type','?')} opts={len(opts)} "
          f"text={q.text[:70]!r}")

# ─────────────────────────────────────────────────────────────
# STAGE 4: First-failure diagnosis
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DIAGNOSIS: FIRST FAILURE POINT")
print(f"{'='*60}")

total_ghost = sum(len(a["unchanged_QUESTION_regions"]) for a in _captured_apply.values())
total_changes = sum(len(a["role_changes"]) for a in _captured_apply.values())
total_vlm_structs = sum(v["structures_produced"] for v in _captured_vlm.values())
pages_with_max_tokens = [pg for pg, v in _captured_vlm.items() if v["finish_reason"] == "MAX_TOKENS"]
pages_with_no_structs = [pg for pg, v in _captured_vlm.items() if v["structures_produced"] == 0]

print(f"""
  VLM ran on {len(_captured_vlm)} pages
  Total VLM structures produced: {total_vlm_structs}
  Pages with MAX_TOKENS (truncated): {pages_with_max_tokens}
  Pages with ZERO VLM structures: {pages_with_no_structs}

  After VLM application:
    Regions whose role was changed by VLM: {total_changes}
    Ghost QUESTION regions NOT addressed by VLM: {total_ghost}
      → These ghost QUESTIONs survive into the graph unchanged

  DocumentStructureGraph:
    Total QUESTION nodes: {_captured_graph.get('role_counts', {}).get('QUESTION', 0)}
    Expected (from VLM structures): {total_vlm_structs} approx
    Excess ghost QUESTION nodes: {_captured_graph.get('role_counts', {}).get('QUESTION', 0) - total_vlm_structs} approx

  FIRST FAILURE POINT:
    _apply_vlm_page_understandings does NOT demote OCR regions
    that the VLM successfully analyzed but did NOT explicitly name.
    These regions keep their deterministic region_type=QUESTION
    from _analyze_region_hypotheses, creating ghost QUESTION nodes
    that flood the graph and pollute extraction.

  ROOT CAUSE CLASSIFICATION:
    VLM authority is ADDITIVE (only adds/changes named regions).
    It should be SELECTIVE (also suppresses unaddressed regions
    on VLM-succeeded pages).
""")

print("Trace complete. See above for full evidence at every stage.")
