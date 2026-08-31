from __future__ import annotations
import os
import uuid
import tempfile
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.core import store
from app.models.schemas import AssessmentStatus, AssessmentResult, AssistantRequest, AssistantResponse
from app.services.document_processor import validate_file, UnsupportedFileError
from app.services.pipeline import run_pipeline
from app.services.assistant_service import process_assistant_message

router = APIRouter(prefix="/api/assessment")

UPLOAD_DIR = os.environ.get(
    "VEDAAI_UPLOAD_DIR",
    os.path.join(tempfile.gettempdir(), "vedaai_uploads"),
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Static Routes First ---

@router.post("/upload")
async def upload(
    question_paper: UploadFile = File(...),
    answer_sheet: UploadFile = File(...),
):
    assessment_id = uuid.uuid4().hex[:12]
    asset_dir = os.path.join(UPLOAD_DIR, assessment_id)
    os.makedirs(asset_dir, exist_ok=True)

    saved = {}
    for role, upload_file in (("question_paper", question_paper), ("answer_sheet", answer_sheet)):
        content = await upload_file.read()
        try:
            ext = validate_file(upload_file.filename, len(content))
        except UnsupportedFileError as e:
            raise HTTPException(status_code=400, detail=str(e))
        path = os.path.join(asset_dir, f"{role}{ext}")
        with open(path, "wb") as f:
            f.write(content)
        saved[role] = path
        saved[f"{role}_ext"] = ext

    store.save_files(
        assessment_id, saved["question_paper"], saved["answer_sheet"],
        saved["question_paper_ext"], saved["answer_sheet_ext"],
    )
    store.set_status(assessment_id, AssessmentStatus(
        assessment_id=assessment_id, state="uploaded", message="Files received", progress=0.05))

    return {"assessment_id": assessment_id}


@router.get("/list")
async def list_all():
    return store.list_assessments()


@router.post("/assistant", response_model=AssistantResponse)
async def global_assistant(req: AssistantRequest):
    return await process_assistant_message(None, req.message, req.question_id)


# --- Dynamic Assessment ID Parameterized Routes ---

@router.post("/{assessment_id}/process")
async def process(assessment_id: str, background_tasks: BackgroundTasks):
    files = store.get_files(assessment_id)
    if not files:
        raise HTTPException(status_code=404, detail="Assessment not found")
    store.set_status(assessment_id, AssessmentStatus(
        assessment_id=assessment_id, state="processing", message="Starting", progress=0.1))
    background_tasks.add_task(run_pipeline, assessment_id)
    return {"assessment_id": assessment_id, "state": "processing"}


@router.get("/{assessment_id}/status", response_model=AssessmentStatus)
async def status(assessment_id: str):
    s = store.get_status(assessment_id)
    if not s:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return s


@router.get("/{assessment_id}/result", response_model=AssessmentResult)
async def result(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    return r


@router.get("/{assessment_id}/results")
async def get_structured_results(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)
        store.save_result(r)
    return r.structured_result


@router.get("/{assessment_id}/review-queue")
async def get_review_queue(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)
        store.save_result(r)
    from app.services.review_service import build_review_queue
    return build_review_queue(r.structured_result.question_results)


@router.get("/{assessment_id}/questions/{question_id}")
async def get_question_detail(assessment_id: str, question_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    q_res = next((q for q in r.questions if q.id == question_id), None)
    if not q_res:
        raise HTTPException(status_code=404, detail="Question not found")
    from app.services.explanation_service import build_question_explanation
    return build_question_explanation(q_res)


class OverrideRequest(BaseModel):
    teacher_marks: float
    criterion_overrides: Optional[dict] = None
    comment: Optional[str] = None
    reason: str = "Teacher manual override"
    reviewer: str = "Teacher"


@router.post("/{assessment_id}/questions/{question_id}/override")
async def override_question_marks(assessment_id: str, question_id: str, req: OverrideRequest):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)
    from app.services.assessment_result_service import apply_teacher_override
    try:
        updated_struct = apply_teacher_override(
            r, question_id, req.teacher_marks, req.criterion_overrides, req.comment, req.reason, req.reviewer
        )
        return updated_struct
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class FinalizeRequest(BaseModel):
    reviewer: str = "Teacher"
    reason: str = "Teacher finalized assessment results"


@router.post("/{assessment_id}/finalize")
async def finalize(assessment_id: str, req: Optional[FinalizeRequest] = None):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)
    from app.services.assessment_result_service import finalize_assessment
    reviewer = req.reviewer if req else "Teacher"
    reason = req.reason if req else "Teacher finalized assessment results"
    return finalize_assessment(r, reviewer, reason)


@router.get("/{assessment_id}/audit-trail")
async def get_audit_trail(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if r.structured_result:
        return r.structured_result.audit_trail
    return r.audit_trail


@router.get("/{assessment_id}/snapshots/{revision_index}")
async def get_assessment_snapshot(assessment_id: str, revision_index: int):
    snapshot_data = store.get_snapshot(assessment_id, revision_index)
    if not snapshot_data:
        raise HTTPException(status_code=404, detail=f"Snapshot for revision {revision_index} not found")
    is_valid, msg = store.verify_snapshot_integrity(snapshot_data)
    res_dict = dict(snapshot_data)
    res_dict["integrity_verified"] = is_valid
    res_dict["integrity_message"] = msg
    return res_dict


@router.get("/{assessment_id}/revisions")
async def get_assessment_revisions(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r or not r.structured_result:
        raise HTTPException(status_code=404, detail="Result not ready")
    return r.structured_result.version_history


@router.get("/{assessment_id}/file/{role}")
async def get_file(assessment_id: str, role: str):
    files = store.get_files(assessment_id)
    if not files or role not in ("question_paper", "answer_sheet"):
        raise HTTPException(status_code=404, detail="Not found")
    path = files[role]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(path)


@router.post("/{assessment_id}/assistant", response_model=AssistantResponse)
async def assistant(assessment_id: str, req: AssistantRequest):
    return await process_assistant_message(assessment_id, req.message, req.question_id)


# -------------------------------------------------------------------------
# STEP 7: STUDENT RESULTS & ASSESSMENT REPORT ENDPOINTS
# -------------------------------------------------------------------------
from fastapi.responses import HTMLResponse


@router.get("/{assessment_id}/student-result")
async def get_student_result(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)

    from app.services.student_result_service import build_student_performance_summary
    return build_student_performance_summary(r.structured_result)


@router.get("/{assessment_id}/report")
async def get_student_report(assessment_id: str, revision_index: Optional[int] = None):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)

    from app.services.report_service import build_student_assessment_report
    return build_student_assessment_report(r.structured_result, revision_index)


@router.get("/{assessment_id}/questions/{question_id}/performance")
async def get_question_performance(assessment_id: str, question_id: str):
    r = store.get_result(assessment_id)
    if not r or not r.structured_result:
        raise HTTPException(status_code=404, detail="Result not ready")

    q = next((item for item in r.structured_result.question_results if item.question_id == question_id), None)
    if not q:
        raise HTTPException(status_code=404, detail=f"Question '{question_id}' not found")

    from app.services.student_result_service import build_question_performance_summary
    return build_question_performance_summary(q)


@router.get("/{assessment_id}/report/export", response_class=HTMLResponse)
async def export_student_report(assessment_id: str, revision_index: Optional[int] = None):
    r = store.get_result(assessment_id)
    if not r or not r.structured_result:
        raise HTTPException(status_code=404, detail="Result not ready")

    from app.services.report_service import build_student_assessment_report, export_report_html
    report = build_student_assessment_report(r.structured_result, revision_index)
    html_content = export_report_html(report)
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/{assessment_id}/insights")
async def get_assessment_insights(assessment_id: str):
    r = store.get_result(assessment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not ready")
    if not r.structured_result:
        from app.services.assessment_result_service import build_structured_assessment_result
        r.structured_result = build_structured_assessment_result(assessment_id, r.questions, r.unmatched_answers)

    from app.services.assessment_insight_service import generate_assessment_insights
    return generate_assessment_insights(r.structured_result)


