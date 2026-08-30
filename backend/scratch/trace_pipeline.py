"""
Deep pipeline trace: VLM output → graph → extraction.
Run AFTER a real assessment completes. Reads the stored result via the store.
Also instruments _apply_vlm_page_understandings and _build_structure_graph
with diagnostic hooks to print exactly what happened to each region.

Usage:
    py scratch\trace_pipeline.py <assessment_id>
"""
import sys, os, json

sys.path.insert(0, ".")
os.environ.setdefault("GEMINI_API_KEY", "x")
os.environ.setdefault("PRIMARY_LLM_PROVIDER", "gemini")
os.environ.setdefault("DOCUMENT_VLM_ENABLED", "true")
os.environ.setdefault("DOCUMENT_VLM_PAGE_UNDERSTANDING", "true")

from app.core import store

assessment_id = sys.argv[1] if len(sys.argv) > 1 else None
if not assessment_id:
    # List available assessments
    print("Usage: py scratch/trace_pipeline.py <assessment_id>")
    sys.exit(1)

result = store.get_result(assessment_id)
if result is None:
    print(f"No result found for assessment_id: {assessment_id}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"PIPELINE TRACE — Assessment {assessment_id}")
print(f"{'='*60}")
print(f"State: {result.state}")
print(f"Total questions extracted: {len(result.questions)}")
print(f"Unmatched answers: {len(result.unmatched_answers)}")

if result.structured_result:
    sr = result.structured_result
    print(f"Structured result: {sr.total_questions} questions, {sr.total_max_marks} max marks")

print(f"\n{'='*60}")
print("EXTRACTED QUESTIONS (all)")
print(f"{'='*60}")
for i, q in enumerate(result.questions):
    opts = q.options if hasattr(q, 'options') else []
    print(f"  [{i+1:02d}] Q{q.number} | type={getattr(q, 'question_type', '?')} | page={q.page} | opts={len(opts)}")
    print(f"       text={q.text[:80]!r}")
    if opts:
        for o in opts:
            print(f"         opt: {o[:60]!r}")
    print(f"       answer_status={q.answer.status} | answer_text={str(q.answer.text or '')[:40]!r}")

if result.structured_result:
    print(f"\n{'='*60}")
    print("STRUCTURED QUESTION RESULTS (API contract)")
    print(f"{'='*60}")
    for sqr in result.structured_result.question_results:
        print(f"  Q{sqr.question_number} | options={sqr.options} | status={sqr.status}")
