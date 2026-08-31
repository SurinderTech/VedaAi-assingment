"""
Storage with disk persistence so server reloads do not wipe assessment states.
"""
from __future__ import annotations
import json
import os
import hashlib
from typing import Dict, Optional, List, Tuple
from app.models.schemas import AssessmentResult, AssessmentStatus

UPLOAD_DIR = os.environ.get(
    "VEDAAI_UPLOAD_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vedaai_uploads")
)
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)
STORE_FILE = os.path.join(UPLOAD_DIR, "store_metadata.json")
SNAPSHOT_DIR = os.path.join(UPLOAD_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

_assessments: Dict[str, AssessmentResult] = {}
_statuses: Dict[str, AssessmentStatus] = {}
_files: Dict[str, dict] = {}
_snapshots: Dict[str, Dict[int, dict]] = {}


def _save_to_disk() -> None:
    try:
        data = {
            "files": _files,
            "statuses": {k: v.model_dump() for k, v in _statuses.items()},
            "assessments": {k: v.model_dump() for k, v in _assessments.items()},
            "snapshots": _snapshots,
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
            for aid, rev_dict in data.get("snapshots", {}).items():
                _snapshots[aid] = {int(k): v for k, v in rev_dict.items()}
    except Exception as e:
        print(f"[Store] Load from disk error: {e}")


_load_from_disk()


def compute_snapshot_hash(snapshot_data: dict) -> str:
    """Computes deterministic SHA-256 hash of snapshot content for tamper detection."""
    payload = {k: v for k, v in snapshot_data.items() if k not in ("snapshot_hash", "integrity_verified")}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_snapshot_integrity(snapshot_data: dict) -> Tuple[bool, str]:
    """Verifies that snapshot data hash matches its stored content hash."""
    stored_hash = snapshot_data.get("snapshot_hash")
    if not stored_hash:
        return False, "No snapshot_hash present"
    computed = compute_snapshot_hash(snapshot_data)
    if computed == stored_hash:
        return True, "Integrity verified: hash matches content"
    return False, f"Integrity check failed: computed {computed[:12]}... != stored {stored_hash[:12]}..."


def save_snapshot(assessment_id: str, revision_index: int, snapshot_data: dict) -> Tuple[str, str]:
    """
    Saves immutable snapshot file for exact assessment revision index.
    Returns (snapshot_file_path, snapshot_hash).
    """
    snapshot_hash = compute_snapshot_hash(snapshot_data)
    snapshot_data["snapshot_hash"] = snapshot_hash
    
    if assessment_id not in _snapshots:
        _snapshots[assessment_id] = {}
        
    _snapshots[assessment_id][revision_index] = snapshot_data
    
    file_name = f"{assessment_id}_rev_{revision_index}.json"
    file_path = os.path.join(SNAPSHOT_DIR, file_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2, default=str)
    except Exception as e:
        print(f"[Store] Save snapshot error: {e}")
        
    _save_to_disk()
    return file_path, snapshot_hash


def get_snapshot(assessment_id: str, revision_index: int) -> Optional[dict]:
    """Retrieves immutable revision snapshot for assessment."""
    if assessment_id in _snapshots and revision_index in _snapshots[assessment_id]:
        return _snapshots[assessment_id][revision_index]
        
    file_name = f"{assessment_id}_rev_{revision_index}.json"
    file_path = os.path.join(SNAPSHOT_DIR, file_name)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if assessment_id not in _snapshots:
                    _snapshots[assessment_id] = {}
                _snapshots[assessment_id][revision_index] = data
                return data
        except Exception as e:
            print(f"[Store] Load snapshot file error: {e}")
            
    return None


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

