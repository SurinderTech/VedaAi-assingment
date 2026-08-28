export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Region {
  page: number;
  bbox: BBox;
}

export type AnswerStatus = "matched" | "unanswered" | "unmatched" | "review_required" | "graded";

export type ReviewStatus = "NOT_REQUIRED" | "PENDING_REVIEW" | "REVIEWED" | "TEACHER_OVERRIDE";

export type EvidenceStatus = "present" | "partially_present" | "missing" | "contradicted" | "uncertain";

export interface MappedAnswer {
  status: AnswerStatus;
  answer_id?: string | null;
  text?: string | null;
  confidence: number;
  method?: string | null;
  regions: Region[];
}

export interface Grading {
  score?: number | null;
  max_score?: number | null;
  strengths: string[];
  missing_points: string[];
  feedback?: string | null;
}

export interface QuestionResult {
  id: string;
  number: string;
  text: string;
  page: number;
  answer: MappedAnswer;
  grading?: Grading | null;
}

export interface UnmatchedAnswer {
  answer_id: string;
  text: string;
  regions: Region[];
  confidence: number;
}

export type ProcessingState =
  | "uploaded" | "processing" | "extracting_questions" | "extracting_answers"
  | "mapping" | "grading" | "completed" | "failed";

export interface AssessmentStatus {
  assessment_id: string;
  state: ProcessingState;
  message?: string | null;
  progress: number;
}

export interface CriterionResult {
  criterion_id: string;
  description: string;
  max_marks: number;
  awarded_marks: number;
  evidence_status: EvidenceStatus;
  confidence: number;
  evidence_text: string;
  provenance: string;
}

export interface TeacherReview {
  review_id: string;
  question_id: string;
  original_ai_marks: number;
  teacher_marks: number;
  reason: string;
  comment?: string | null;
  reviewer?: string | null;
  timestamp: string;
  changed: boolean;
}

export interface AuditEvent {
  event_id: string;
  timestamp: string;
  assessment_id: string;
  question_id?: string | null;
  event_type: string;
  previous_value?: any;
  new_value?: any;
  source: string;
  reason?: string | null;
}

export interface AssessmentRevision {
  revision_id: string;
  revision_index: number;
  timestamp: string;
  final_awarded_marks: number;
  percentage: number;
  finalized_by: string;
  reason: string;
  snapshot_hash?: string | null;
  snapshot_file?: string | null;
}

export interface StructuredQuestionResult {
  question_id: string;
  question_number: string;
  question_text: string;
  max_marks: number;
  answer_id?: string | null;
  answer_text: string;
  answer_pages: number[];
  answer_regions: any[];
  status: AnswerStatus;
  awarded_marks: number;
  original_ai_marks: number;
  teacher_adjusted_marks?: number | null;
  evaluation_confidence: number;
  needs_review: boolean;
  criterion_results: CriterionResult[];
  evidence_summary: string[];
  feedback: string;
  mapping_provenance?: string | null;
  grading_provenance?: string | null;
  escalation_reason?: string | null;
  review_status: ReviewStatus;
  teacher_review?: TeacherReview | null;
}

export interface StructuredAssessmentResult {
  assessment_id: string;
  assessment_status: "IN_REVIEW" | "FINALIZED";
  revision_index: number;
  total_questions: number;
  answered_questions: number;
  unanswered_questions: number;
  unmatched_answers_count: number;
  total_max_marks: number;
  ai_awarded_marks: number;
  teacher_adjusted_marks?: number | null;
  final_awarded_marks: number;
  percentage: number;
  overall_confidence: number;
  questions_needing_review: number;
  question_results: StructuredQuestionResult[];
  review_summary: Record<string, any>;
  grading_statistics: Record<string, any>;
  audit_trail: AuditEvent[];
  version_history: AssessmentRevision[];
  created_at: string;
  updated_at: string;
}

export interface AssessmentResult {
  assessment_id: string;
  state: ProcessingState;
  questions: QuestionResult[];
  unmatched_answers: UnmatchedAnswer[];
  question_paper_pages: number;
  answer_sheet_pages: number;
  answer_sheet_page_sizes: number[][];
  answer_sheet_is_pdf: boolean;
  question_paper_url?: string | null;
  answer_sheet_url?: string | null;
  structured_result?: StructuredAssessmentResult | null;
  audit_trail?: AuditEvent[];
}

export interface StudentPerformanceSummary {
  assessment_id: string;
  total_max_marks: number;
  final_awarded_marks: number;
  percentage: number;
  overall_confidence: number;
  answered_questions: number;
  unanswered_questions: number;
  questions_needing_review: number;
  performance_band: string;
}

export interface CriterionPerformanceSummary {
  criterion_id: string;
  description: string;
  max_marks: number;
  awarded_marks: number;
  evidence_status: string;
  confidence: number;
  provenance: string;
}

export interface QuestionPerformanceSummary {
  question_id: string;
  question_number: string;
  question_text: string;
  max_marks: number;
  final_awarded_marks: number;
  percentage: number;
  status: string;
  feedback: string;
  strengths: string[];
  improvement_points: string[];
  review_status: string;
  source_regions: any[];
  criteria_summary: CriterionPerformanceSummary[];
}

export interface StudentAssessmentReport {
  assessment_id: string;
  assessment_status: string;
  final_score: number;
  total_max_marks: number;
  percentage: number;
  performance_summary: StudentPerformanceSummary;
  question_results: QuestionPerformanceSummary[];
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  feedback: string;
  generated_at: string;
  report_version: number;
}

export interface AssessmentInsight {
  insight_id: string;
  type: "STRENGTH" | "WEAKNESS" | "ERROR_PATTERN" | "REVIEW_PRIORITY" | "GENERAL";
  title: string;
  summary: string;
  question_ids: string[];
  evidence_refs: string[];
  confidence: number;
  source: string;
}

export interface QuestionInsight {
  question_id: string;
  question_number: string;
  strengths: string[];
  improvement_areas: string[];
  error_patterns: string[];
  evidence_refs: string[];
  source_regions: any[];
  confidence: number;
}

export interface AssessmentInsights {
  assessment_id: string;
  final_awarded_marks: number;
  total_max_marks: number;
  percentage: number;
  answered_questions: number;
  unanswered_questions: number;
  unmatched_answers_count: number;
  questions_needing_review: number;
  strengths: string[];
  areas_needing_attention: string[];
  error_patterns: AssessmentInsight[];
  review_priorities: AssessmentInsight[];
  question_insights: QuestionInsight[];
  generated_at: string;
}


