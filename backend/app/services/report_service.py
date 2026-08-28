"""
Assessment Report Service (Step 7).
Assembles complete StudentAssessmentReport and clean export representations (HTML/PDF).
STRICT SAFETY RULES:
- Consumes finalized assessment results and revision snapshots without mutation.
- Does not modify grading decisions or snapshot files.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from app.models.schemas import (
    StructuredAssessmentResult,
    StudentAssessmentReport,
    QuestionPerformanceSummary,
)
from app.services.student_result_service import (
    build_student_performance_summary,
    build_question_performance_summary,
)
from app.services.student_feedback_service import generate_student_report_feedback
from app.core import store


def build_student_assessment_report(
    result: StructuredAssessmentResult,
    revision_index: Optional[int] = None,
) -> StudentAssessmentReport:
    """
    Constructs unified StudentAssessmentReport for current or requested historical revision.
    """
    target_result = result
    if revision_index is not None and revision_index > 0:
        snapshot_dict = store.get_snapshot(result.assessment_id, revision_index)
        if snapshot_dict:
            target_result = StructuredAssessmentResult.model_validate(snapshot_dict)

    summary = build_student_performance_summary(target_result)
    q_summaries: List[QuestionPerformanceSummary] = [
        build_question_performance_summary(q) for q in target_result.question_results
    ]

    fb_dict = generate_student_report_feedback(target_result)
    now_iso = datetime.now(timezone.utc).isoformat()

    return StudentAssessmentReport(
        assessment_id=target_result.assessment_id,
        assessment_status=target_result.assessment_status,
        final_score=target_result.final_awarded_marks,
        total_max_marks=target_result.total_max_marks,
        percentage=summary.percentage,
        performance_summary=summary,
        question_results=q_summaries,
        strengths=fb_dict.get("strengths", []),
        weaknesses=fb_dict.get("weaknesses", []),
        recommendations=fb_dict.get("recommendations", []),
        feedback=fb_dict.get("summary", ""),
        generated_at=now_iso,
        report_version=target_result.revision_index,
    )


def export_report_html(report: StudentAssessmentReport) -> str:
    """Generates clean exportable HTML assessment report representation."""
    questions_html = ""
    for q in report.question_results:
        strengths_li = "".join(f"<li>{s}</li>" for s in q.strengths)
        improvements_li = "".join(f"<li>{i}</li>" for i in q.improvement_points)
        questions_html += f"""
        <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #e2e8f0; border-radius: 12px; background: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 16px; font-weight: bold; color: #0f172a;">Q{q.question_number}: {q.question_text}</h3>
                <span style="font-weight: bold; color: #4f46e5; font-size: 14px;">{q.final_awarded_marks} / {q.max_marks} pts ({q.percentage}%)</span>
            </div>
            <p style="margin: 4px 0; font-size: 13px; color: #334155;">{q.feedback}</p>
            {f'<div style="margin-top: 8px; font-size: 12px; color: #166534;"><b>Strengths:</b><ul>{strengths_li}</ul></div>' if strengths_li else ''}
            {f'<div style="margin-top: 8px; font-size: 12px; color: #991b1b;"><b>Improvement Areas:</b><ul>{improvements_li}</ul></div>' if improvements_li else ''}
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Assessment Report - {report.assessment_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #0f172a; background: #f8fafc; }}
        .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #cbd5e1; padding-bottom: 20px; }}
        .score-card {{ background: #0f172a; color: white; padding: 20px; border-radius: 16px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 12px; color: #0f172a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; font-size: 24px; color: #0f172a;">VedaAI Student Assessment Report</h1>
        <p style="margin: 6px 0 0 0; color: #64748b; font-size: 13px;">Assessment ID: {report.assessment_id} | Revision #{report.report_version} | Status: {report.assessment_status}</p>
    </div>

    <div class="score-card">
        <div>
            <div style="font-size: 13px; text-transform: uppercase; color: #94a3b8;">Final Awarded Score</div>
            <div style="font-size: 32px; font-weight: 900; margin-top: 4px;">{report.final_score} <span style="font-size: 16px; color: #cbd5e1;">/ {report.total_max_marks}</span></div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 28px; font-weight: 900; color: #34d399;">{report.percentage}%</div>
            <div style="font-size: 12px; color: #e2e8f0; margin-top: 2px;">Band: {report.performance_summary.performance_band}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Performance Summary</div>
        <p style="font-size: 14px; color: #334155; line-height: 1.6;">{report.feedback}</p>
    </div>

    <div class="section">
        <div class="section-title">Question Performance Breakdown</div>
        {questions_html}
    </div>
</body>
</html>"""
