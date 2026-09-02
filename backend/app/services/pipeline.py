"""
Assessment Pipeline — 100% Pure Multimodal VLM/LLM Document Intelligence.

Orchestrates the end-to-end evaluation flow:
  1. Render Question Paper & Answer Sheet to high-fidelity images
  2. VLM Visual Question Understanding (extract assessable questions, subquestions, MCQs, bboxes)
  3. VLM Visual Answer Understanding (transcribe student handwriting, MCQs, tables, bboxes)
  4. VLM/LLM Semantic Question ↔ Answer Mapping & Unanswered Detection
  5. LLM Evidence-Based Grading & Feedback
  6. Package results with coordinates, bboxes, and persistent storage
"""
from __future__ import annotations

import asyncio
from typing import Dict, Any, List
from app.core import store
from app.models.schemas import (
    AssessmentResult,
    AssessmentStatus,
    QuestionResult,
    Question,
    AnswerCandidate,
)
from app.services.document_processor import render_document_images
from app.services.question_extractor import extract_questions_vlm
from app.services.answer_extractor import extract_answers_vlm
from app.services.mapping_engine import map_answers_vlm
from app.services.grading_service import generate_grading
from app.services.assessment_result_service import build_structured_assessment_result

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
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="failed",
                message=f"⏱️ Processing timed out (>{PIPELINE_TIMEOUT_SECONDS}s). Please try uploading smaller files.",
                progress=0,
            ),
        )


async def _run_pipeline_inner(assessment_id: str) -> None:
    files = store.get_files(assessment_id)
    if not files:
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="failed",
                message="Uploaded files not found in store.",
                progress=0,
            ),
        )
        return

    try:
        # ── STEP 1: Render Question Paper Images ─────────────────────────────
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="extracting_questions",
                message="📄 VLM visually inspecting Question Paper structure...",
                progress=0.15,
            ),
        )
        qp_pages, qp_sizes, qp_images = await asyncio.to_thread(
            render_document_images, files["question_paper"], files["question_paper_ext"]
        )

        # ── STEP 2: Pure VLM Question Paper Extraction ───────────────────────
        questions: List[Question] = await extract_questions_vlm(qp_images)
        if not questions:
            print("[Pipeline] VLM questions extraction returned empty list on 1st attempt, retrying VLM...")
            questions = await extract_questions_vlm(qp_images)

        if not questions:
            raise RuntimeError(
                "VLM Question Extraction failed: 0 questions extracted from the question paper. "
                "This indicates an unreadable document or VLM provider failure. "
                "The pipeline will not masquerade provider failure as '0 questions found'."
            )

        print(f"[Pipeline] Extracted {len(questions)} assessable questions via VLM.")

        # ── STEP 3: Render Answer Sheet Images ───────────────────────────────
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="extracting_answers",
                message="🖊️ VLM visually reading student answer sheet & handwriting...",
                progress=0.45,
            ),
        )
        as_pages, as_sizes, as_images = await asyncio.to_thread(
            render_document_images, files["answer_sheet"], files["answer_sheet_ext"]
        )

        # ── STEP 4: Pure VLM Answer Sheet Extraction ─────────────────────────
        answers: List[AnswerCandidate] = await extract_answers_vlm(as_images)
        print(f"[Pipeline] Extracted {len(answers)} student answer candidates via VLM.")

        # ── STEP 5: Pure VLM/LLM Semantic Mapping ────────────────────────────
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="mapping",
                message="🔗 VLM/LLM semantically mapping answers to questions...",
                progress=0.70,
            ),
        )
        mapped, unmatched = await map_answers_vlm(questions, answers)

        # ── STEP 6: AI Grading & Feedback ────────────────────────────────────
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="grading",
                message="⭐ Generating AI evaluation, rubric scoring & teacher feedback...",
                progress=0.88,
            ),
        )

        # Grade all questions concurrently
        gradings = await asyncio.gather(*[generate_grading(q, mapped[q.id]) for q in questions])

        question_results = [
            QuestionResult(
                id=q.id,
                number=q.number,
                text=q.text,
                page=q.page,
                answer=mapped[q.id],
                grading=g,
                section=q.section,
                options=q.options,
            )
            for q, g in zip(questions, gradings)
        ]

        # ── STEP 7: Package Structured Result & Coordinates ──────────────────
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
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="completed",
                message="✅ Assessment evaluation complete",
                progress=1.0,
            ),
        )
        print(f"[Pipeline] Assessment {assessment_id} completed successfully.")

    except Exception as e:
        print(f"[PipelineError] Assessment {assessment_id} failed: {e}")
        store.set_status(
            assessment_id,
            AssessmentStatus(
                assessment_id=assessment_id,
                state="failed",
                message=f"Processing failed: {e}",
                progress=0,
            ),
        )