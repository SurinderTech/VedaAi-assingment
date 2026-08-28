"""
Audit Trail Logging Service.
Logs immutable audit events for state changes, teacher reviews, mark overrides, feedback edits, and finalization.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, List
from app.models.schemas import AuditEvent


def create_audit_event(
    assessment_id: str,
    event_type: str,
    question_id: Optional[str] = None,
    previous_value: Optional[Any] = None,
    new_value: Optional[Any] = None,
    source: str = "system",
    reason: Optional[str] = None,
) -> AuditEvent:
    """Creates a structured AuditEvent object with ISO timestamp."""
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = f"aud_{uuid.uuid4().hex[:12]}"
    return AuditEvent(
        event_id=event_id,
        timestamp=now_iso,
        assessment_id=assessment_id,
        question_id=question_id,
        event_type=event_type,
        previous_value=previous_value,
        new_value=new_value,
        source=source,
        reason=reason,
    )
