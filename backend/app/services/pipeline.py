from __future__ import annotations
import asyncio
import gc
from app.core import store
from app.models.schemas import (
    AssessmentResult, AssessmentStatus, QuestionResult,
)
from app.services.document_processor import process_document
from app.services.page_intelligence import analyze_pages
from app.services.question_extractor import extract_questions
from app.services.answer_extractor import extract_answers
from app.services.mapping_engine import map_answers
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
        # ── STEP 1: Question Paper OCR ───────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_questions",
            message="📄 Reading question paper (OCR + AI vision)...", progress=0.10))
        qp_res = await asyncio.to_thread(
            process_document, files["question_paper"], files["question_paper_ext"], False
        )
        qp_blocks = qp_res[0]
        qp_pages = qp_res[1]
        qp_sizes = qp_res[2]
        qp_images = qp_res[3] if len(qp_res) > 3 else None
        page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(qp_sizes)} if qp_sizes else None

        # ── STEP 2: VLM understanding of question paper ──────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_questions",
            message=f"🧠 AI analyzing {qp_pages} page(s) of question paper — this takes ~{qp_pages * 8}s...", progress=0.20))

        from app.services.document_understanding_service import DocumentUnderstandingService
        doc_understanding_res = await asyncio.to_thread(
            DocumentUnderstandingService().process_document,
            qp_blocks,
            f"doc_{assessment_id}",
            page_sizes_dict,
            qp_images,
        )

        # ── STEP 3: Extract questions ────────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_questions",
            message="🔍 Extracting questions from document structure...", progress=0.40))
        questions = await extract_questions(
            qp_blocks, doc_understanding_result=doc_understanding_res, page_sizes=page_sizes_dict
        )

        # ── FREE QP memory before answer-sheet OCR ───────────────────────────
        # qp_images holds raw PIL Images for every page — can be 50-200MB on
        # a 512 MB Render instance. Delete explicitly before the 2nd OCR call.
        del qp_images, qp_res, doc_understanding_res
        gc.collect()
        print(f"[Pipeline] QP memory freed. Starting answer-sheet processing...")

        # ── STEP 4: Answer Sheet OCR ─────────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_answers",
            message="🖊️ Reading handwritten answer sheet...", progress=0.45))
        as_res = await asyncio.to_thread(
            process_document, files["answer_sheet"], files["answer_sheet_ext"], True
        )
        as_blocks = as_res[0]
        as_pages = as_res[1]
        as_sizes = as_res[2]
        as_images = as_res[3] if len(as_res) > 3 else None
        as_page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(as_sizes)} if as_sizes else None

        # ── STEP 4B: VLM understanding of the answer sheet ────────────────────
        # The answer sheet gets the SAME genuine visual-understanding treatment as the
        # question paper (STEP 2). Without this, answer extraction falls back to pure
        # OCR-text regex matching, which cannot reliably read handwriting or associate
        # a table cell's answer with its row's question number — this is the primary
        # cause of "No answer detected" showing up for every question.
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_answers",
            message=f"🧠 AI analyzing {as_pages} page(s) of answer sheet — this takes ~{as_pages * 8}s...", progress=0.50))

        from app.services.document_understanding_service import DocumentUnderstandingService
        as_doc_understanding_res = await asyncio.to_thread(
            DocumentUnderstandingService().process_document,
            as_blocks,
            f"as_doc_{assessment_id}",
            as_page_sizes_dict,
            as_images,
        )

        # ── STEP 5: Extract answers ──────────────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_answers",
            message="🧠 AI reading handwritten answers...", progress=0.55))
        page_types, metadata_pages = analyze_pages(as_blocks, as_pages)
        answers = await asyncio.to_thread(
            extract_answers, as_blocks, metadata_pages, as_doc_understanding_res
        )

        # Free AS images before mapping/grading
        del as_images, as_res, as_doc_understanding_res
        gc.collect()

        # ── STEP 6: Map answers to questions ─────────────────────────────────
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="mapping",
            message="🔗 Mapping answers to questions...", progress=0.75))
        mapped, unmatched = await map_answers(questions, answers)

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