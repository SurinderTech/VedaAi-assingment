"""
Page Intelligence Engine.

Analyzes every page of the answer sheet PDF/document independently using multi-signal analysis:
- Calculates metadata_likelihood, ocr_density, anchor_count, blank_likelihood.
- Classifies pages into METADATA, ANSWER_CONTENT, CONTINUATION, BLANK, MIXED, UNKNOWN.
- Keywords (e.g. "Name", "Roll No") are weighted supporting signals, not sole page discards.
- Preserves raw coordinates and page signal metrics for downstream extraction.
"""
from __future__ import annotations

import re
from typing import List, Dict, Set, Tuple, Optional
from app.models.schemas import Block, PageAnalysis, PageClassification, QuestionAnchor

METADATA_KEYWORD_PATTERNS = [
    r"\buniversity\b",
    r"\banswer\s*booklet\b",
    r"\bstudent\s*name\b",
    r"\broll\s*no\b",
    r"\bregistration\s*no\b",
    r"\benrollment\s*no\b",
    r"\bhall\s*ticket\b",
    r"\binvigilator\b",
    r"\bseat\s*no\b",
    r"\bcenter\s*code\b",
    r"\bcourse\b",
    r"\bsemester\b",
]

HEADER_NOISE_RE = re.compile(
    r"^\s*(?:SECTION\s*-\s*[A-Z0-9]+\s*\(continued\)|ANSWERSHEET|Page\s*\d+\s*(?:of|/)\s*\d+)",
    re.IGNORECASE,
)


def analyze_pages(
    blocks: List[Block],
    num_pages: int,
    anchors_by_page: Optional[Dict[int, List[QuestionAnchor]]] = None,
) -> Tuple[Dict[int, PageAnalysis], Set[int]]:
    """
    Multi-signal page intelligence analyzer.
    
    Returns:
      page_analyses: Dict[page_num, PageAnalysis]
      metadata_pages: Set of page indices classified as pure METADATA cover pages.
    """
    if anchors_by_page is None:
        anchors_by_page = {}

    blocks_by_page: Dict[int, List[Block]] = {p: [] for p in range(1, num_pages + 1)}
    for b in blocks:
        if b.page in blocks_by_page:
            blocks_by_page[b.page].append(b)

    page_analyses: Dict[int, PageAnalysis] = {}
    metadata_pages: Set[int] = set()

    for page_num in range(1, num_pages + 1):
        p_blocks = blocks_by_page[page_num]
        p_anchors = anchors_by_page.get(page_num, [])

        if not p_blocks:
            page_analyses[page_num] = PageAnalysis(
                page=page_num,
                classification="BLANK",
                confidence=0.99,
                metadata_likelihood=0.0,
                ocr_density=0.0,
                anchors=[],
            )
            continue

        full_text = " ".join(b.text.lower() for b in p_blocks)
        total_chars = len(full_text.strip())
        ocr_density = round(len(p_blocks) / 50.0, 3)

        # 1. Multi-signal metadata likelihood calculation
        meta_kw_matches = sum(1 for pat in METADATA_KEYWORD_PATTERNS if re.search(pat, full_text))
        metadata_likelihood = min(1.0, round(meta_kw_matches / 4.0, 2))

        has_cover_header = any(
            t in full_text for t in ["answer booklet", "semester examination", "student name", "roll no"]
        )

        student_anchor_count = sum(1 for a in p_anchors if a.role == "student_question_anchor")
        question_ref_count = sum(1 for a in p_anchors if a.role == "question_reference")

        # 2. Multi-Signal Page Classification Decision
        if total_chars < 15 and len(p_anchors) == 0:
            classification: PageClassification = "BLANK"
            confidence = 0.95

        elif has_cover_header and student_anchor_count == 0 and metadata_likelihood >= 0.50:
            classification = "METADATA"
            confidence = 0.95
            metadata_pages.add(page_num)

        elif metadata_likelihood >= 0.30 and (student_anchor_count > 0 or question_ref_count > 0):
            # Page has metadata fields mixed with question references or student answers
            classification = "MIXED"
            confidence = 0.90

        elif student_anchor_count > 0:
            classification = "ANSWER_CONTENT"
            confidence = 0.95

        else:
            prev_analysis = page_analyses.get(page_num - 1)
            if prev_analysis and prev_analysis.classification in ("ANSWER_CONTENT", "CONTINUATION", "MIXED"):
                classification = "CONTINUATION"
                confidence = 0.85
            else:
                classification = "ANSWER_CONTENT"
                confidence = 0.70

        page_analyses[page_num] = PageAnalysis(
            page=page_num,
            classification=classification,
            confidence=confidence,
            metadata_likelihood=metadata_likelihood,
            ocr_density=ocr_density,
            anchors=p_anchors,
        )

    return page_analyses, metadata_pages
