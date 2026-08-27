"""
Storage with disk persistence so server reloads do not wipe assessment states.
"""
from __future__ import annotations
import json
import os
import tempfile
from typing import Dict, Optional, List
from app.models.schemas import AssessmentResult, AssessmentStatus

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "vedaai_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
STORE_FILE = os.path.join(UPLOAD_DIR, "store_metadata.json")

_assessments: Dict[str, AssessmentResult] = {}
_statuses: Dict[str, AssessmentStatus] = {}
_files: Dict[str, dict] = {}


def _save_to_disk() -> None:
    try:
        data = {
            "files": _files,
            "statuses": {k: v.model_dump() for k, v in _statuses.items()},
            "assessments": {k: v.model_dump() for k, v in _assessments.items()},
        }
        with open(STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
    except Exception as e:
        print(f"[Store] Save to disk error: {e}")


def _load_from_disk() -> None:
    if not os.path.exists(STORE_FILE):
        return
    try:
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _files.update(data.get("files", {}))
            for k, v in data.get("statuses", {}).items():
                _statuses[k] = AssessmentStatus(**v)
            for k, v in data.get("assessments", {}).items():
                _assessments[k] = AssessmentResult(**v)
    except Exception as e:
        print(f"[Store] Load from disk error: {e}")


_load_from_disk()


def save_files(assessment_id: str, question_paper: str, answer_sheet: str,
                question_paper_ext: str = "", answer_sheet_ext: str = "") -> None:
    _files[assessment_id] = {
        "question_paper": question_paper,
        "answer_sheet": answer_sheet,
        "question_paper_ext": question_paper_ext,
        "answer_sheet_ext": answer_sheet_ext,
    }
    _save_to_disk()


def get_files(assessment_id: str) -> Optional[dict]:
    return _files.get(assessment_id)


def set_status(assessment_id: str, status: AssessmentStatus) -> None:
    _statuses[assessment_id] = status
    _save_to_disk()


def get_status(assessment_id: str) -> Optional[AssessmentStatus]:
    return _statuses.get(assessment_id)


def save_result(result: AssessmentResult) -> None:
    _assessments[result.assessment_id] = result
    _save_to_disk()


def get_result(assessment_id: str) -> Optional[AssessmentResult]:
    return _assessments.get(assessment_id)


def list_assessments() -> List[dict]:
    res = []
    for aid, s in _statuses.items():
        r = _assessments.get(aid)
        q_count = len(r.questions) if r else 0
        matched = sum(1 for q in r.questions if q.answer.status == "matched") if r else 0
        unanswered = sum(1 for q in r.questions if q.answer.status == "unanswered") if r else 0
        review = sum(1 for q in r.questions if q.answer.status == "review_required") if r else 0
        res.append({
            "assessment_id": aid,
            "state": s.state,
            "progress": s.progress,
            "message": s.message,
            "question_count": q_count,
            "matched_count": matched,
            "unanswered_count": unanswered,
            "review_count": review,
        })
    return res

