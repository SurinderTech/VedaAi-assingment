from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Region(BaseModel):
    page: int
    bbox: BBox


class Block(BaseModel):
    """Normalized low-level unit produced by OCR / native PDF text extraction."""
    id: str
    text: str
    confidence: float
    bbox: BBox
    page: int
    type: Literal["text", "line"] = "text"
    source: Literal["native_pdf", "ocr"] = "ocr"


class Question(BaseModel):
    id: str
    number: str
    text: str
    page: int
    bbox: Optional[BBox] = None
    order_index: int


class AnswerCandidate(BaseModel):
    answer_id: str
    question_number: Optional[str] = None
    text: str
    regions: List[Region]
    order_index: int


AnswerStatus = Literal["matched", "unanswered", "unmatched", "review_required"]


class MappedAnswer(BaseModel):
    status: AnswerStatus
    answer_id: Optional[str] = None
    text: Optional[str] = None
    confidence: float = 0.0
    method: Optional[str] = None
    regions: List[Region] = []


class Grading(BaseModel):
    score: Optional[float] = None
    max_score: Optional[float] = None
    strengths: List[str] = []
    missing_points: List[str] = []
    feedback: Optional[str] = None


class QuestionResult(BaseModel):
    id: str
    number: str
    text: str
    page: int
    answer: MappedAnswer
    grading: Optional[Grading] = None


class UnmatchedAnswer(BaseModel):
    answer_id: str
    text: str
    regions: List[Region]
    confidence: float


ProcessingState = Literal[
    "uploaded",
    "processing",
    "extracting_questions",
    "extracting_answers",
    "mapping",
    "grading",
    "completed",
    "failed",
]


class AssessmentStatus(BaseModel):
    assessment_id: str
    state: ProcessingState
    message: Optional[str] = None
    progress: float = 0.0


class AssessmentResult(BaseModel):
    assessment_id: str
    state: ProcessingState
    questions: List[QuestionResult] = []
    unmatched_answers: List[UnmatchedAnswer] = []
    question_paper_pages: int = 0
    answer_sheet_pages: int = 0
    # (width, height) in pixels of the coordinate space that answer bboxes
    # were computed in, per page (1-indexed page N = sizes[N-1]). The
    # frontend scales bbox -> screen coords against these, per plan §26.
    answer_sheet_page_sizes: List[List[int]] = []
    answer_sheet_is_pdf: bool = False
    question_paper_url: Optional[str] = None
    answer_sheet_url: Optional[str] = None
