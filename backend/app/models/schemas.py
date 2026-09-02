from __future__ import annotations
from typing import List, Optional, Literal, Dict, Set, Any
from pydantic import BaseModel


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Region(BaseModel):
    page: int
    bbox: BBox


BlockModality = Literal[
    "handwritten",
    "printed",
    "mixed",
    "unknown",
]

BlockRole = Literal[
    "question_reference",
    "student_question_anchor",
    "student_answer",
    "page_metadata",
    "instruction",
    "header_footer",
    "noise",
    "visual_element",
    "unknown",
]


class Block(BaseModel):
    """Normalized low-level unit produced by OCR / native PDF text extraction."""
    id: str
    text: str
    confidence: float
    bbox: BBox
    page: int
    type: Literal["text", "line"] = "text"
    source: Literal["native_pdf", "ocr"] = "ocr"
    modality: BlockModality = "unknown"
    role: BlockRole = "unknown"


QuestionType = Literal[
    "SHORT_ANSWER",
    "LONG_ANSWER",
    "MCQ",
    "SUBQUESTION",
    "NUMERICAL",
    "DIAGRAM",
    "TABLE",
    "UNKNOWN",
]


class ExtractedOption(BaseModel):
    option_id: str
    question_id: str
    label: str
    text: str
    source_region_ids: List[str] = []
    source_regions: List[Region] = []
    extraction_confidence: float = 1.0
    verification_state: str = "UNVERIFIED"


class Question(BaseModel):
    id: str
    number: str
    text: str
    page: int
    bbox: Optional[BBox] = None
    order_index: int = 0
    section: Optional[str] = None
    options: List[str] = []
    source_region_ids: List[str] = []
    source_regions: List[Region] = []
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    parent_question_id: Optional[str] = None
    question_type: QuestionType = "UNKNOWN"
    extracted_options: List[ExtractedOption] = []
    extraction_confidence: float = 1.0
    verification_state: str = "UNVERIFIED"
    max_marks: Optional[float] = 2.0
    # Nested subquestions (e.g. Q3(i), Q3(ii)) — stored here, NOT in flat question list
    subquestions: Optional[List["Question"]] = None


# Resolve forward reference for subquestions: Optional[List["Question"]]
Question.model_rebuild()


PageClassification = Literal[
    "METADATA",
    "ANSWER_CONTENT",
    "CONTINUATION",
    "BLANK",
    "MIXED",
    "UNKNOWN",
]


class QuestionAnchor(BaseModel):
    anchor: str
    original_text: str
    role: Literal["student_question_anchor", "question_reference"] = "student_question_anchor"
    page: int
    bbox: BBox
    confidence: float = 0.9
    reading_order: int = 0


class PageAnalysis(BaseModel):
    page: int
    classification: PageClassification
    confidence: float = 0.9
    metadata_likelihood: float = 0.0
    ocr_density: float = 0.0
    anchors: List[QuestionAnchor] = []


class AnswerRegion(BaseModel):
    answer_id: str
    question_anchor: Optional[str] = None
    pages: List[int]
    regions: List[Region]
    text: str
    blocks: List[Block] = []
    reading_order: int = 0
    is_continuation: bool = False
    confidence: float = 0.9
    ocr_text: str = ""
    vlm_text: str = ""
    selected_text: str = ""
    text_source: str = "OCR"
    grounding_status: str = "UNGROUNDED"
    grounded_ocr_region_ids: List[str] = []
    vlm_region_id: Optional[str] = None
    vlm_confidence: float = 0.0
    answer_to: Optional[str] = None
    answer_to_question_number: Optional[str] = None
    answer_to_confidence: float = 0.0
    provenance: Dict[str, Any] = {}
    review_required: bool = False
    needs_review: bool = False


class StructuredAnswerSheet(BaseModel):
    num_pages: int
    page_sizes: List[List[int]] = []
    page_analyses: Dict[int, PageAnalysis] = {}
    question_anchors: List[QuestionAnchor] = []
    question_references: List[QuestionAnchor] = []
    answer_regions: List[AnswerRegion] = []
    unanchored_regions: List[AnswerRegion] = []
    metadata_blocks: List[Block] = []
    printed_question_blocks: List[Block] = []
    noise_blocks: List[Block] = []


class AnswerCandidate(BaseModel):
    answer_id: str
    question_number: Optional[str] = None
    text: str
    regions: List[Region]
    order_index: int



AnswerStatus = Literal["matched", "unanswered", "unmatched", "review_required", "graded"]


class MappedAnswer(BaseModel):
    status: AnswerStatus
    answer_id: Optional[str] = None
    text: Optional[str] = None
    confidence: float = 0.0
    method: Optional[str] = None
    regions: List[Region] = []
    anchor_score: float = 0.0
    semantic_score: float = 0.0
    structural_score: float = 0.0
    spatial_score: float = 0.0
    order_score: float = 0.0
    w_anchor: float = 0.40
    w_semantic: float = 0.30
    w_structural: float = 0.15
    w_spatial: float = 0.10
    w_order: float = 0.05
    raw_final_score: float = 0.0
    conflict_penalty: float = 1.00
    final_score: float = 0.0
    best_candidate_score: float = 0.0
    second_best_candidate_score: float = 0.0
    score_margin: float = 0.0
    conflict_detected: bool = False
    needs_review: bool = False
    evidence_summary: Optional[str] = None
    raw_region: Optional[AnswerRegion] = None


QuestionType = Literal[
    "definition",
    "short_conceptual",
    "long_conceptual",
    "comparison",
    "explanation",
    "process",
    "numerical",
    "mathematical_derivation",
    "mcq",
    "one_word",
    "true_false",
    "fill_blank",
    "code",
    "algorithm",
    "diagram",
    "diagram_explanation",
    "table",
    "mixed",
    "unknown",
]

AnswerContentType = Literal[
    "text",
    "short_text",
    "long_text",
    "number",
    "formula",
    "mathematical_work",
    "code",
    "diagram",
    "table",
    "mcq_selection",
    "mixed",
    "visual_only",
    "unknown",
]

CriterionStatus = Literal["present", "partially_present", "missing", "contradicted", "uncertain"]


class QuestionRequirementSpec(BaseModel):
    expected_answer_type: QuestionType = "unknown"
    max_marks: Optional[float] = None
    marks_confidence: str = "high"
    required_concepts: List[str] = []
    required_operations: List[str] = []
    has_diagram_requirement: bool = False
    has_numerical_requirement: bool = False
    has_code_requirement: bool = False
    evaluation_criteria_descriptions: List[str] = []


class RubricCriterion(BaseModel):
    id: str
    description: str
    max_marks: float
    weight: float = 1.0


class Rubric(BaseModel):
    answer_type: QuestionType = "unknown"
    total_max_marks: float = 2.0
    criteria: List[RubricCriterion] = []


EvidenceProvenance = Literal[
    "local",
    "llm",
    "fused_agreement",
    "fused_resolution",
    "conflict_flagged",
]


class CriterionEvidence(BaseModel):
    criterion_id: str
    description: str
    status: CriterionStatus = "uncertain"
    evidence_text: Optional[str] = None
    confidence: float = 0.0
    awarded_marks: float = 0.0
    max_marks: float = 0.0
    notes: Optional[str] = None
    provenance: Optional[EvidenceProvenance] = "local"


RoutingDecision = Literal[
    "LOCAL_CLEAR",
    "LOCAL_CLEAR_WITH_HIGH_CONFIDENCE",
    "LLM_REQUIRED",
    "LLM_RECOMMENDED",
    "REVIEW_REQUIRED",
]


class GradingResult(BaseModel):
    question_id: str
    answer_id: Optional[str] = None
    max_marks: float = 2.0
    awarded_marks: float = 0.0
    confidence: float = 0.0
    status: AnswerStatus = "graded"
    needs_review: bool = False
    answer_type: QuestionType = "unknown"
    content_type: AnswerContentType = "unknown"
    criteria: List[CriterionEvidence] = []
    correct_evidence: List[str] = []
    missing_evidence: List[str] = []
    incorrect_evidence: List[str] = []
    partial_evidence: List[str] = []
    uncertain_evidence: List[str] = []
    semantic_score: float = 0.0
    factual_score: float = 0.0
    completeness_score: float = 0.0
    mathematical_score: float = 0.0
    visual_score: float = 0.0
    code_score: float = 0.0
    evaluation_method: str = "local"
    routing_decision: Optional[RoutingDecision] = None
    escalation_reason: Optional[str] = None
    llm_used: bool = False
    llm_provider: Optional[str] = None
    total_questions: int = 1
    local_evaluations: int = 1
    llm_evaluations: int = 0
    llm_calls_avoided: int = 1
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_tokens_saved: int = 0
    token_provenance: str = "estimated"
    llm_failure_count: int = 0
    feedback: str = ""


class Grading(BaseModel):
    score: Optional[float] = None
    max_score: Optional[float] = None
    strengths: List[str] = []
    missing_points: List[str] = []
    feedback: Optional[str] = None
    result_details: Optional[GradingResult] = None


class QuestionResult(BaseModel):
    id: str
    number: str
    text: str
    page: int
    answer: MappedAnswer
    grading: Optional[Grading] = None
    section: Optional[str] = None
    options: List[str] = []


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


ReviewStatus = Literal[
    "NOT_REQUIRED",
    "PENDING_REVIEW",
    "REVIEWED",
    "TEACHER_OVERRIDE",
]


class CriterionResult(BaseModel):
    criterion_id: str
    description: str
    max_marks: float
    awarded_marks: float
    evidence_status: CriterionStatus = "uncertain"
    evidence_text: Optional[str] = None
    confidence: float = 0.0
    provenance: Optional[EvidenceProvenance] = "local"
    needs_review: bool = False
    source_regions: List[Dict[str, Any]] = []


class TeacherReview(BaseModel):
    review_id: str
    question_id: str
    original_ai_marks: float
    teacher_marks: float
    reason: str
    comment: Optional[str] = None
    reviewer: Optional[str] = "Teacher"
    timestamp: str = ""
    changed: bool = True


class AuditEvent(BaseModel):
    event_id: str
    timestamp: str = ""
    assessment_id: str
    question_id: Optional[str] = None
    event_type: str
    previous_value: Optional[Any] = None
    new_value: Optional[Any] = None
    source: str = "system"
    reason: Optional[str] = None


class AssessmentRevision(BaseModel):
    revision_id: str
    revision_index: int
    timestamp: str = ""
    final_awarded_marks: float
    percentage: float
    finalized_by: str = "Teacher"
    reason: str = "Finalization"
    snapshot_hash: Optional[str] = None
    snapshot_file: Optional[str] = None


class StructuredQuestionResult(BaseModel):
    question_id: str
    question_number: str
    question_text: str
    max_marks: float
    answer_id: Optional[str] = None
    answer_text: str = ""
    answer_pages: List[int] = []
    answer_regions: List[Dict[str, Any]] = []
    status: AnswerStatus = "graded"
    awarded_marks: float = 0.0
    original_ai_marks: float = 0.0
    teacher_adjusted_marks: Optional[float] = None
    evaluation_confidence: float = 0.0
    needs_review: bool = False
    criterion_results: List[CriterionResult] = []
    evidence_summary: List[str] = []
    feedback: str = ""
    mapping_provenance: Optional[str] = "explicit_question_anchor"
    grading_provenance: Optional[str] = "local"
    escalation_reason: Optional[str] = None
    review_status: ReviewStatus = "NOT_REQUIRED"
    teacher_review: Optional[TeacherReview] = None
    # VLM-extracted MCQ options preserved end-to-end from document understanding
    options: List[str] = []
    # Fix 5: Full structured semantic data preserved through API
    question_type: str = "UNKNOWN"  # MCQ, SHORT_ANSWER, LONG_ANSWER, SUBQUESTION, etc.
    parent_question_id: Optional[str] = None
    page_number: int = 0
    semantic_state: str = "UNKNOWN"  # CONFIDENT, PARTIAL, AMBIGUOUS, UNKNOWN
    source_region_ids: List[str] = []  # Provenance: region IDs from document graph
    extraction_confidence: float = 1.0
    extracted_options: List[Dict[str, Any]] = []  # Structured options with label/text/confidence


class StructuredAssessmentResult(BaseModel):
    assessment_id: str
    assessment_status: Literal["IN_REVIEW", "FINALIZED"] = "IN_REVIEW"
    revision_index: int = 1
    total_questions: int = 0
    answered_questions: int = 0
    unanswered_questions: int = 0
    unmatched_answers_count: int = 0
    total_max_marks: float = 0.0
    ai_awarded_marks: float = 0.0
    teacher_adjusted_marks: Optional[float] = None
    final_awarded_marks: float = 0.0
    percentage: float = 0.0
    overall_confidence: float = 0.0
    questions_needing_review: int = 0
    question_results: List[StructuredQuestionResult] = []
    review_summary: Dict[str, Any] = {}
    grading_statistics: Dict[str, Any] = {}
    audit_trail: List[AuditEvent] = []
    version_history: List[AssessmentRevision] = []
    created_at: str = ""
    updated_at: str = ""


class AssessmentResult(BaseModel):
    assessment_id: str
    state: ProcessingState
    questions: List[QuestionResult] = []
    unmatched_answers: List[UnmatchedAnswer] = []
    question_paper_pages: int = 0
    answer_sheet_pages: int = 0
    answer_sheet_page_sizes: List[List[int]] = []
    answer_sheet_is_pdf: bool = False
    question_paper_url: Optional[str] = None
    answer_sheet_url: Optional[str] = None
    structured_result: Optional[StructuredAssessmentResult] = None
    audit_trail: List[AuditEvent] = []


class AssistantRequest(BaseModel):
    message: str
    question_id: Optional[str] = None


class AssistantResponse(BaseModel):
    reply: str
    attention_questions: List[str] = []
    unanswered_questions: List[str] = []
    review_questions: List[str] = []


class StudentPerformanceSummary(BaseModel):
    assessment_id: str
    total_max_marks: float
    final_awarded_marks: float
    percentage: float
    overall_confidence: float
    answered_questions: int
    unanswered_questions: int
    questions_needing_review: int
    performance_band: str = "Proficient"


class CriterionPerformanceSummary(BaseModel):
    criterion_id: str
    description: str
    max_marks: float
    awarded_marks: float
    evidence_status: str
    confidence: float
    provenance: str = "local"


class QuestionPerformanceSummary(BaseModel):
    question_id: str
    question_number: str
    question_text: str
    max_marks: float
    final_awarded_marks: float
    percentage: float
    status: str = "graded"
    feedback: str = ""
    strengths: List[str] = []
    improvement_points: List[str] = []
    review_status: str = "NOT_REQUIRED"
    source_regions: List[Dict[str, Any]] = []
    criteria_summary: List[CriterionPerformanceSummary] = []


class StudentAssessmentReport(BaseModel):
    assessment_id: str
    assessment_status: str = "FINALIZED"
    final_score: float
    total_max_marks: float
    percentage: float
    performance_summary: StudentPerformanceSummary
    question_results: List[QuestionPerformanceSummary] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommendations: List[str] = []
    feedback: str = ""
    generated_at: str = ""
    report_version: int = 1


class AssessmentInsight(BaseModel):
    insight_id: str
    type: Literal["STRENGTH", "WEAKNESS", "ERROR_PATTERN", "REVIEW_PRIORITY", "GENERAL"] = "GENERAL"
    title: str
    summary: str
    question_ids: List[str] = []
    evidence_refs: List[str] = []
    confidence: float = 0.9
    source: str = "evidence_engine"


class QuestionInsight(BaseModel):
    question_id: str
    question_number: str
    strengths: List[str] = []
    improvement_areas: List[str] = []
    error_patterns: List[str] = []
    evidence_refs: List[str] = []
    source_regions: List[Dict[str, Any]] = []
    confidence: float = 0.9


class AssessmentInsights(BaseModel):
    assessment_id: str
    final_awarded_marks: float
    total_max_marks: float
    percentage: float
    answered_questions: int
    unanswered_questions: int
    unmatched_answers_count: int
    questions_needing_review: int
    strengths: List[str] = []
    areas_needing_attention: List[str] = []
    error_patterns: List[AssessmentInsight] = []
    review_priorities: List[AssessmentInsight] = []
    question_insights: List[QuestionInsight] = []
    generated_at: str = ""


StructuredQuestionResult.model_rebuild()
StructuredAssessmentResult.model_rebuild()
AssessmentResult.model_rebuild()
StudentPerformanceSummary.model_rebuild()
QuestionPerformanceSummary.model_rebuild()
StudentAssessmentReport.model_rebuild()
AssessmentInsight.model_rebuild()
QuestionInsight.model_rebuild()
AssessmentInsights.model_rebuild()


# ============================================================
# STEP 11A — Universal Document Understanding Foundation Schemas
# ============================================================

DocumentRegionType = Literal[
    "HEADER",
    "FOOTER",
    "METADATA",
    "INSTRUCTION",
    "SECTION_HEADER",
    "QUESTION",
    "SUBQUESTION",
    "OPTION",
    "TABLE",
    "TABLE_CELL",
    "DIAGRAM",
    "FIGURE",
    "CAPTION",
    "ANSWER_SPACE",
    "ANSWER_REGION",
    "HANDWRITING",
    "FORM_FIELD",
    "PARAGRAPH",
    "LIST",
    "SIGNATURE",
    "UNKNOWN",
]

RelationshipType = Literal[
    "follows",
    "contains",
    "belongs_to",
    "continuation_of",
    "same_structure_as",
    "adjacent_to",
    "visually_grouped_with",
    "uncertain_relation",
    "option_of",
    "subquestion_of",
    "section_member",
    "associated_visual",
    "answer_to",
    "caption_of",
]

DocumentPurpose = Literal[
    "QUESTION_PAPER",
    "INSTRUCTIONS",
    "COVER",
    "ANSWER_KEY",
    "REFERENCE",
    "UNKNOWN",
]


class RegionManifestItem(BaseModel):
    region_id: str
    page: int
    bbox: BBox
    ocr_text: str
    initial_hypothesis: str = "UNKNOWN"
    neighbors: List[str] = []


class RegionManifest(BaseModel):
    page_number: int
    page_width: float
    page_height: float
    regions: List[RegionManifestItem] = []


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: RelationshipType
    confidence: float = 1.0
    evidence_sources: List[str] = []
    semantic_state: Literal["CONFIDENT", "PARTIAL", "AMBIGUOUS", "UNKNOWN", "UNRESOLVED", "CONFLICTING"] = "CONFIDENT"


class GraphNode(BaseModel):
    region_id: str
    role: DocumentRegionType
    text: str
    page: int
    bbox: BBox
    confidence: float = 1.0
    semantic_state: Literal["CONFIDENT", "PARTIAL", "AMBIGUOUS", "UNKNOWN", "UNRESOLVED", "CONFLICTING"] = "CONFIDENT"


class DocumentStructureGraph(BaseModel):
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    document_purpose: DocumentPurpose = "UNKNOWN"
    page_roles: Dict[int, str] = {}
    graph_semantic_state: Literal["CONFIDENT", "PARTIAL", "AMBIGUOUS", "UNKNOWN", "UNRESOLVED", "CONFLICTING"] = "CONFIDENT"


SignalType = Literal[
    "numbering_pattern",
    "question_interrogative",
    "spatial_position",
    "surrounding_regions",
    "semantic_signal",
    "indentation",
    "vertical_horizontal_proximity",
    "repeated_layout_pattern",
    "text_density",
    "punctuation",
    "option_formatting",
    "section_formatting",
    "heading_formatting",
    "table_geometry",
    "page_position",
    "continuation_relationship",
]


class DocumentEvidence(BaseModel):
    signal_type: str
    description: str
    weight: float = 1.0
    score: float = 1.0
    metadata: Dict[str, Any] = {}


class StructureHypothesis(BaseModel):
    region_id: str
    hypothesized_type: DocumentRegionType
    confidence: float
    source: str = "parser"
    evidence: List[DocumentEvidence] = []


class RegionRelationship(BaseModel):
    source_region_id: str
    target_region_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    evidence: List[DocumentEvidence] = []


VerificationState = Literal["VERIFIED", "CONFLICTED", "UNCERTAIN", "UNVERIFIED"]


class CostAccounting(BaseModel):
    pages_considered: int = 0
    pages_sent: int = 0
    regions_considered: int = 0
    regions_sent: int = 0
    vlm_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_high_confidence_count: int = 0
    cache_hits: int = 0


class RegionVerificationSpec(BaseModel):
    region_id: str
    page: int
    bbox: BBox
    current_type: DocumentRegionType = "UNKNOWN"
    neighbors: List[str] = []
    ocr_text: str = ""
    reason_for_verification: str = "ambiguity"


class VLMHypothesis(BaseModel):
    region_id: str
    proposed_type: DocumentRegionType = "UNKNOWN"
    confidence: float = 0.5
    reasoning: str = ""
    relationships: List[Dict[str, Any]] = []
    uncertainty: float = 0.0
    conflict_indicators: List[str] = []


class VisualVerificationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str = "NOT_CONFIGURED"
    is_available: bool = False
    model_name: str = "vlm_default"
    vlm_hypotheses: List[VLMHypothesis] = []
    verified_relationships: List[RegionRelationship] = []
    rejected_vlm_relationships: List[Dict[str, Any]] = []
    cost_accounting: CostAccounting = CostAccounting()
    error_message: Optional[str] = None



class DocumentRegion(BaseModel):
    region_id: str
    page: int
    text: str
    bbox: BBox
    region_type: DocumentRegionType = "UNKNOWN"
    source: str = "ocr"
    confidence: float = 1.0
    evidence: List[DocumentEvidence] = []
    relationships: List[RegionRelationship] = []
    uncertainty: float = 0.0
    classification_conflict: bool = False
    conflicting_hypotheses: List[StructureHypothesis] = []
    embedding: Optional[List[float]] = None
    parent_region_id: Optional[str] = None
    child_region_ids: List[str] = []
    verification_state: VerificationState = "UNVERIFIED"
    vlm_hypothesis: Optional[StructureHypothesis] = None
    metadata: Dict[str, Any] = {}


class DocumentPage(BaseModel):
    page_number: int
    width: float = 0.0
    height: float = 0.0
    regions: List[DocumentRegion] = []
    reading_order: List[str] = []


class DocumentObservation(BaseModel):
    doc_id: str
    pages: List[DocumentPage] = []
    raw_blocks: List[Block] = []


class DocumentUnderstandingResult(BaseModel):
    document_id: str
    pages: List[DocumentPage] = []
    regions: List[DocumentRegion] = []
    relationships: List[RegionRelationship] = []
    conflicts: List[Dict[str, Any]] = []
    vlm_status: str = "NOT_CONFIGURED"
    verification_summary: Dict[str, Any] = {}
    cost_accounting: Optional[CostAccounting] = None
    metadata: Dict[str, Any] = {}
    structure_graph: Optional[DocumentStructureGraph] = None
    vlm_page_understandings: List[Any] = []  # List[VLMPageUnderstanding] (forward ref)
    document_purpose: str = "UNKNOWN"
    page_roles: Dict[int, str] = {}


# ============================================================
# STEP 11C — Intelligent Question Extraction Container & Audit Schemas
# ============================================================

class ExtractedSection(BaseModel):
    section_id: str
    title: str
    page: int
    bbox: Optional[BBox] = None
    source_region_ids: List[str] = []
    question_ids: List[str] = []


class RejectionRecord(BaseModel):
    region_id: str
    ocr_text: str
    classification: str
    confidence: float
    reason: str
    evidence_refs: List[str] = []


class ExtractionAudit(BaseModel):
    candidate_count: int = 0
    accepted_question_count: int = 0
    rejected_count: int = 0
    uncertain_count: int = 0
    option_count: int = 0
    section_count: int = 0
    multi_region_question_count: int = 0
    multi_page_question_count: int = 0
    duplicate_rejected: int = 0  # Fix 2: tracks document-scoped duplicate QUESTION nodes
    conflicts: List[str] = []
    rejection_reasons: List[RejectionRecord] = []
    invariant_violations: List[str] = []


ExtractedQuestion = Question


class VLMStructureItem(BaseModel):
    """One structural element identified by the VLM's page understanding.

    region_ids are optional because a meaningful visual structure may be discovered
    from the page image even when no OCR block already encodes that semantic unit.
    bbox is the authoritative visual geometry for such visual-only structures and is
    later grounded back to OCR evidence when matching region IDs exist.
    """
    region_ids: List[str] = []
    grounded_region_ids: List[str] = []
    grounding_status: str = "UNGROUNDED"
    grounded_text: str = ""
    bbox: Optional[BBox] = None
    role: DocumentRegionType = "UNKNOWN"
    display_number: Optional[str] = None
    display_label: Optional[str] = None
    vlm_text: str = ""
    reasoning: str = ""
    confidence: float = 0.5


class VLMRelationshipItem(BaseModel):
    """One relationship identified by the VLM between structural elements."""
    source_ids: List[str] = []
    target_ids: List[str] = []
    relationship_type: str = "belongs_to"
    confidence: float = 0.5


class VLMPageUnderstanding(BaseModel):
    """The VLM's independent understanding of a single document page."""
    page_number: int
    page_purpose: str = "UNKNOWN"
    document_purpose: str = "UNKNOWN"
    structures: List[VLMStructureItem] = []
    relationships: List[VLMRelationshipItem] = []
    raw_response: str = ""
    vlm_model: str = ""
    image_sent: bool = False
    image_dimensions: Optional[List[float]] = None
    image_bytes: int = 0
    base64_chars: int = 0
    ocr_blocks_sent: int = 0
    prompt_chars: int = 0
    vlm_attempt: bool = True
    structure_source: str = "DETERMINISTIC_FALLBACK"
    vlm_provider: str = "N/A"
    vlm_result: str = "NOT_ATTEMPTED"
    finish_reason: str = "N/A"
    semantic_completeness: str = "UNKNOWN"  # COMPLETE | PARTIAL | UNKNOWN
    retry_count: int = 0
    fallback_provider: str = "N/A"
    structures_produced: int = 0
    relationships_produced: int = 0



class DocumentQuestionExtractionResult(BaseModel):
    document_id: str
    questions: List[ExtractedQuestion] = []
    sections: List[ExtractedSection] = []
    uncertain_candidates: List[ExtractedQuestion] = []
    audit: ExtractionAudit = ExtractionAudit()
    structure_graph: Optional[DocumentStructureGraph] = None
    fallback_used: bool = False
    invariant_violations: List[str] = []


VLMStructureItem.model_rebuild()
VLMRelationshipItem.model_rebuild()
VLMPageUnderstanding.model_rebuild()
DocumentEvidence.model_rebuild()
StructureHypothesis.model_rebuild()
RegionRelationship.model_rebuild()
DocumentRegion.model_rebuild()
DocumentPage.model_rebuild()
DocumentObservation.model_rebuild()
DocumentUnderstandingResult.model_rebuild()
CostAccounting.model_rebuild()
RegionVerificationSpec.model_rebuild()
VLMHypothesis.model_rebuild()
VisualVerificationResponse.model_rebuild()
ExtractedOption.model_rebuild()
ExtractedSection.model_rebuild()
RejectionRecord.model_rebuild()
ExtractionAudit.model_rebuild()
DocumentQuestionExtractionResult.model_rebuild()