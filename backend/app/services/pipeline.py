from __future__ import annotations
import asyncio
import gc
from app.core import store
from app.models.schemas import (
    AssessmentResult, AssessmentStatus, QuestionResult,
)
from app.services.document_processor import process_document
from app.services.page_intelligence import analyze_pages
from app.services.question_extractor import extract_questions, extract_questions_vlm
from app.services.answer_extractor import extract_answers, extract_answers_vlm
from app.services.mapping_engine import map_answers, map_answers_vlm
from app.services.grading_service import generate_grading

# Hard timeout: if processing takes longer than this, fail gracefully
PIPELINE_TIMEOUT_SECONDS = 360  # 6 minutes max


async def run_pipeline(assessment_id: str) -> None:
    """Wrapper that enforces a hard timeout on the full pipeline."""
    try:
        await asyncio.wait_for(
            _run_pipeline_inner(assessment_id),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(f"[Pipeline] TIMEOUT after {PIPELINE_TIMEOUT_SECONDS}s for assessment {assessment_id}")
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id,
            state="failed",
            message=(
                f"⏱️ Processing took too long (>{PIPELINE_TIMEOUT_SECONDS}s). "
                "Try uploading a smaller document or fewer pages."
            ),
            progress=0,
        ))


async def _run_pipeline_inner(assessment_id: str) -> None:
    files = store.get_files(assessment_id)
    if not files:
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="failed", message="Files not found", progress=0))
        return

    try:
        # ── STEP 1: Process Question Paper ───────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_questions",
            message="📄 Gemini VLM reading question paper images directly...", progress=0.15))
        qp_res = await asyncio.to_thread(
            process_document, files["question_paper"], files["question_paper_ext"], True
        )
        qp_blocks, qp_pages, qp_sizes, qp_images = qp_res

        # ── STEP 2: 100% VLM Question Extraction ──────────────────────────────
        if qp_images:
            questions = await extract_questions_vlm(qp_images)
            if not questions:
                print("[Pipeline] VLM questions extraction returned empty list on 1st attempt, retrying VLM...")
                questions = await extract_questions_vlm(qp_images)
        else:
            questions = await extract_questions(qp_blocks)

        print(f"[Pipeline] Extracted {len(questions)} questions via Gemini VLM.")

        # ── STEP 3: Process Answer Sheet ─────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_answers",
            message="🖊️ Gemini VLM reading handwritten student answer sheet images directly...", progress=0.45))
        as_res = await asyncio.to_thread(
            process_document, files["answer_sheet"], files["answer_sheet_ext"], True
        )
        as_blocks, as_pages, as_sizes, as_images = as_res

        # ── STEP 4: 100% VLM Answer Extraction ───────────────────────────────
        if as_images:
            answers = await extract_answers_vlm(as_images)
        else:
            page_types, metadata_pages = analyze_pages(as_blocks, as_pages)
            answers = extract_answers(as_blocks, metadata_pages)

        print(f"[Pipeline] Extracted {len(answers)} student answer candidates via Gemini VLM.")

        # ── STEP 5: VLM/LLM Mapping ──────────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="mapping",
            message="🔗 VLM/LLM mapping student answers to questions...", progress=0.75))
        mapped, unmatched = await map_answers_vlm(questions, answers)

        # ── STEP 7: Grade all questions ──────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="grading",
            message="⭐ Generating AI score & feedback...", progress=0.90))

        # Parallelize grading for all questions concurrently
        gradings = await asyncio.gather(*[generate_grading(q, mapped[q.id]) for q in questions])

        question_results = [
            QuestionResult(
                id=q.id, number=q.number, text=q.text, page=q.page,
                answer=mapped[q.id], grading=g, section=q.section, options=q.options
            )
            for q, g in zip(questions, gradings)
        ]

        from app.services.assessment_result_service import build_structured_assessment_result
        structured_res = build_structured_assessment_result(assessment_id, question_results, unmatched)

        result = AssessmentResult(
            assessment_id=assessment_id,
            state="completed",
            questions=question_results,
            unmatched_answers=unmatched,
            question_paper_pages=qp_pages,
            answer_sheet_pages=as_pages,
            answer_sheet_page_sizes=[[int(w), int(h)] for (w, h) in as_sizes],
            answer_sheet_is_pdf=(files["answer_sheet_ext"] == ".pdf"),
            structured_result=structured_res,
            audit_trail=structured_res.audit_trail,
        )
        store.save_result(result)
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="completed",
            message="✅ Done", progress=1.0))

    except Exception as e:  # noqa: BLE001
        print(f"[PipelineError] Assessment {assessment_id} failed: {e}")
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="failed",
            message=f"Processing failed: {e}", progress=0))