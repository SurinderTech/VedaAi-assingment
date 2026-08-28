from __future__ import annotations
import asyncio
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


async def run_pipeline(assessment_id: str) -> None:
    files = store.get_files(assessment_id)
    if not files:
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="failed", message="Files not found", progress=0))
        return

    try:
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_questions",
            message="Reading question paper", progress=0.15))
        qp_blocks, qp_pages, qp_sizes = await asyncio.to_thread(
            process_document, files["question_paper"], files["question_paper_ext"], False
        )
        page_sizes_dict = {i + 1: [float(w), float(h)] for i, (w, h) in enumerate(qp_sizes)} if qp_sizes else None

        from app.services.document_understanding_service import DocumentUnderstandingService
        doc_understanding_res = await asyncio.to_thread(
            DocumentUnderstandingService().process_document,
            qp_blocks,
            f"doc_{assessment_id}",
            page_sizes_dict,
        )
        questions = await extract_questions(qp_blocks, doc_understanding_result=doc_understanding_res, page_sizes=page_sizes_dict)

        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_answers",
            message="Reading handwritten answers", progress=0.45))
        # Answer sheet ALWAYS uses force_ocr=True so bboxes are in image pixel coordinates
        as_blocks, as_pages, as_sizes = await asyncio.to_thread(
            process_document, files["answer_sheet"], files["answer_sheet_ext"], True
        )
        page_types, metadata_pages = analyze_pages(as_blocks, as_pages)
        answers = await asyncio.to_thread(extract_answers, as_blocks, metadata_pages)

        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="mapping",
            message="Mapping answers to questions", progress=0.75))
        mapped, unmatched = await map_answers(questions, answers)

        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="grading",
            message="Generating AI score & feedback", progress=0.90))

        # Parallelize grading for all questions concurrently with asyncio.gather
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
            message="Done", progress=1.0))

    except Exception as e:  # noqa: BLE001
        print(f"[PipelineError] Assessment {assessment_id} failed: {e}")
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="failed",
            message=f"Processing failed: {e}", progress=0))
