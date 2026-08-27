from __future__ import annotations
import os
import uuid
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.core import store
from app.models.schemas import AssessmentStatus, AssessmentResult, AssistantRequest, AssistantResponse
from app.services.document_processor import validate_file, UnsupportedFileError
from app.services.pipeline import run_pipeline
from app.services.assistant_service import process_assistant_message

router = APIRouter(prefix="/api/assessment")

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "vedaai_uploads")
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
