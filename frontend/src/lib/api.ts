import { AssessmentResult, AssessmentStatus } from "@/types/assessment";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadAssessment(questionPaper: File, answerSheet: File): Promise<string> {
  const form = new FormData();
  form.append("question_paper", questionPaper);
  form.append("answer_sheet", answerSheet);
  const res = await fetch(`${API_URL}/api/assessment/upload`, { method: "POST", body: form });
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

export function fileUrl(assessmentId: string, role: "question_paper" | "answer_sheet"): string {
  return `${API_URL}/api/assessment/${assessmentId}/file/${role}`;
}
