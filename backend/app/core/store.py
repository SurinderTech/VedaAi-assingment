"""
Plan section 37: no database. Temporary in-memory storage is sufficient
for a single-session assessment tool. Everything here is lost on
process restart -- acceptable for the assignment's scope.
"""
from __future__ import annotations
from typing import Dict, Optional
from app.models.schemas import AssessmentResult, AssessmentStatus

_assessments: Dict[str, AssessmentResult] = {}
_statuses: Dict[str, AssessmentStatus] = {}
_files: Dict[str, dict] = {}  # assessment_id -> {"question_paper": path, "answer_sheet": path, ...}


def save_files(assessment_id: str, question_paper: str, answer_sheet: str,
                question_paper_ext: str = "", answer_sheet_ext: str = "") -> None:
    _files[assessment_id] = {
        "question_paper": question_paper,
        "answer_sheet": answer_sheet,
        "question_paper_ext": question_paper_ext,
        "answer_sheet_ext": answer_sheet_ext,
    }


def get_files(assessment_id: str) -> Optional[dict]:
    return _files.get(assessment_id)


def set_status(assessment_id: str, status: AssessmentStatus) -> None:
    _statuses[assessment_id] = status


def get_status(assessment_id: str) -> Optional[AssessmentStatus]:
    return _statuses.get(assessment_id)


def save_result(result: AssessmentResult) -> None:
    _assessments[result.assessment_id] = result


def get_result(assessment_id: str) -> Optional[AssessmentResult]:
    return _assessments.get(assessment_id)
