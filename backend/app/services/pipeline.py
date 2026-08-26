from __future__ import annotations
from app.core import store
from app.models.schemas import (
    AssessmentResult, AssessmentStatus, QuestionResult,
)
from app.services.document_processor import process_document
from app.services.question_extractor import extract_questions
from app.services.answer_extractor import extract_answers
from app.services.mapping_engine import map_answers


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
        qp_blocks, qp_pages, _ = process_document(files["question_paper"], files["question_paper_ext"], force_ocr=False)
        questions = extract_questions(qp_blocks)

        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="extracting_answers",
            message="Reading handwritten answers", progress=0.45))
        # Answer sheet ALWAYS uses force_ocr=True so bboxes are in image pixel coordinates
        as_blocks, as_pages, as_sizes = process_document(files["answer_sheet"], files["answer_sheet_ext"], force_ocr=True)
        answers = extract_answers(as_blocks)

        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="mapping",
            message="Mapping answers to questions", progress=0.75))
        mapped, unmatched = await map_answers(questions, answers)

        question_results = [
            QuestionResult(
                id=q.id, number=q.number, text=q.text, page=q.page,
                answer=mapped[q.id], grading=None,
            )
            for q in questions
        ]

        result = AssessmentResult(
            assessment_id=assessment_id,
            state="completed",
            questions=question_results,
            unmatched_answers=unmatched,
            question_paper_pages=qp_pages,
            answer_sheet_pages=as_pages,
            answer_sheet_page_sizes=[[int(w), int(h)] for (w, h) in as_sizes],
            answer_sheet_is_pdf=(files["answer_sheet_ext"] == ".pdf"),
        )
        store.save_result(result)
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="completed",
            message="Done", progress=1.0))

    except Exception as e:  # noqa: BLE001
        store.set_status(assessment_id, AssessmentStatus(
            assessment_id=assessment_id, state="failed",
            message=f"Processing failed: {e}", progress=0))
