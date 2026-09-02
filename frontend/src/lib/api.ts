import {
  AssessmentResult,
  AssessmentStatus,
  StructuredAssessmentResult,
  StructuredQuestionResult,
  AuditEvent,
  AssessmentRevision,
  StudentPerformanceSummary,
  QuestionPerformanceSummary,
  StudentAssessmentReport,
  AssessmentInsights,
} from "@/types/assessment";

export function getApiUrl(): string {
  if (typeof window !== "undefined") {
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }
  }
  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (envUrl && envUrl.trim() && !envUrl.includes("modal.run")) {
    return envUrl.trim().replace(/\/+$/, "");
  }
  return "http://localhost:8000";
}

const API_URL = "http://localhost:8000";

export async function uploadAssessment(questionPaper: File, answerSheet: File): Promise<string> {
  const form = new FormData();
  form.append("question_paper", questionPaper);
  form.append("answer_sheet", answerSheet);
  const res = await fetch(`${getApiUrl()}/api/assessment/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  const data = await res.json();
  return data.assessment_id as string;
}

export async function startProcessing(assessmentId: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/process`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to start processing: ${res.status}`);
}

export async function getStatus(assessmentId: string): Promise<AssessmentStatus> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/status`);
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
  return res.json();
}

export async function getResult(assessmentId: string): Promise<AssessmentResult> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/result`);
  if (!res.ok) throw new Error(`Result fetch failed: ${res.status}`);
  return res.json();
}

export async function getStructuredResult(assessmentId: string): Promise<StructuredAssessmentResult> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/results`);
  if (!res.ok) throw new Error(`Structured result fetch failed: ${res.status}`);
  return res.json();
}

export async function getReviewQueue(assessmentId: string): Promise<StructuredQuestionResult[]> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/review-queue`);
  if (!res.ok) throw new Error(`Review queue fetch failed: ${res.status}`);
  return res.json();
}

export async function getQuestionDetail(assessmentId: string, questionId: string): Promise<StructuredQuestionResult> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/questions/${questionId}`);
  if (!res.ok) throw new Error(`Question detail fetch failed: ${res.status}`);
  return res.json();
}

export interface OverridePayload {
  teacher_marks: number;
  criterion_overrides?: Record<string, number>;
  comment?: string;
  reason?: string;
  reviewer?: string;
}

export async function overrideQuestionMarks(
  assessmentId: string,
  questionId: string,
  payload: OverridePayload
): Promise<StructuredAssessmentResult> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/questions/${questionId}/override`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
    throw new Error(errData.detail || `Override failed: ${res.status}`);
  }
  return res.json();
}

export interface FinalizePayload {
  reviewer?: string;
  reason?: string;
}

export async function finalizeAssessment(
  assessmentId: string,
  payload?: FinalizePayload
): Promise<StructuredAssessmentResult> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/finalize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw new Error(`Finalization failed: ${res.status}`);
  return res.json();
}

export async function getAuditTrail(assessmentId: string): Promise<AuditEvent[]> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/audit-trail`);
  if (!res.ok) throw new Error(`Audit trail fetch failed: ${res.status}`);
  return res.json();
}

export async function getRevisionSnapshot(assessmentId: string, revisionIndex: number): Promise<any> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/snapshots/${revisionIndex}`);
  if (!res.ok) throw new Error(`Snapshot fetch failed: ${res.status}`);
  return res.json();
}

export async function getRevisions(assessmentId: string): Promise<AssessmentRevision[]> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/revisions`);
  if (!res.ok) throw new Error(`Revisions fetch failed: ${res.status}`);
  return res.json();
}

export async function listAssessments(): Promise<any[]> {
  try {
    const res = await fetch(`${API_URL}/api/assessment/list`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export interface AssistantResponse {
  reply: string;
  attention_questions: string[];
  unanswered_questions: string[];
  review_questions: string[];
}

export async function askAssistant(
  assessmentId: string | null,
  message: string,
  questionId?: string
): Promise<AssistantResponse> {
  const endpoint = assessmentId
    ? `${API_URL}/api/assessment/${assessmentId}/assistant`
    : `${API_URL}/api/assessment/assistant`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, question_id: questionId }),
  });

  if (!res.ok) {
    throw new Error(`Assistant request failed: ${res.status}`);
  }
  return res.json();
}

export function fileUrl(assessmentId: string, role: "question_paper" | "answer_sheet"): string {
  return `${API_URL}/api/assessment/${assessmentId}/file/${role}`;
}

export async function getStudentResult(assessmentId: string): Promise<StudentPerformanceSummary> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/student-result`);
  if (!res.ok) throw new Error(`Student result fetch failed: ${res.status}`);
  return res.json();
}

export async function getStudentReport(
  assessmentId: string,
  revisionIndex?: number
): Promise<StudentAssessmentReport> {
  const url = revisionIndex
    ? `${API_URL}/api/assessment/${assessmentId}/report?revision_index=${revisionIndex}`
    : `${API_URL}/api/assessment/${assessmentId}/report`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Student report fetch failed: ${res.status}`);
  return res.json();
}

export async function getQuestionPerformance(
  assessmentId: string,
  questionId: string
): Promise<QuestionPerformanceSummary> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/questions/${questionId}/performance`);
  if (!res.ok) throw new Error(`Question performance fetch failed: ${res.status}`);
  return res.json();
}

export function exportStudentReportUrl(assessmentId: string, revisionIndex?: number): string {
  return revisionIndex
    ? `${API_URL}/api/assessment/${assessmentId}/report/export?revision_index=${revisionIndex}`
    : `${API_URL}/api/assessment/${assessmentId}/report/export`;
}

export async function getAssessmentInsights(assessmentId: string): Promise<AssessmentInsights> {
  const res = await fetch(`${API_URL}/api/assessment/${assessmentId}/insights`);
  if (!res.ok) throw new Error(`Assessment insights fetch failed: ${res.status}`);
  return res.json();
}


