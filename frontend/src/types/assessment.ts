export interface BBox { x: number; y: number; width: number; height: number; }
export interface Region { page: number; bbox: BBox; }

export type AnswerStatus = "matched" | "unanswered" | "unmatched" | "review_required";

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

export interface AssessmentResult {
  assessment_id: string;
  state: ProcessingState;
  questions: QuestionResult[];
  unmatched_answers: UnmatchedAnswer[];
  question_paper_pages: number;
  answer_sheet_pages: number;
  answer_sheet_page_sizes: number[][];
  answer_sheet_is_pdf: boolean;
}
