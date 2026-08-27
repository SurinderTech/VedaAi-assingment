"""
Page Intelligence Engine.

Analyzes every page of the answer sheet PDF/document before mapping:
- Detects metadata/cover pages (Student Name, Roll No, University Header, Total Marks, etc.)
- Detects answer content pages and continuation pages.
- Provides page classification and metadata page penalty flags so cover pages are NEVER mapped as answers.
"""
from __future__ import annotations
import re
from typing import List, Dict, Set, Tuple
from app.models.schemas import Block

METADATA_KEYWORDS = [
    "university", "semester examination", "answer booklet", "student name", "roll no",
    "registration no", "course", "semester", "subject", "total marks", "examination",
    "enrollment no", "hall ticket", "invigilator", "seat no", "center code"
]

ANCHOR_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*(\([a-zA-Z1-9]\)|[a-zA-Z](?=[\s\.:\)]|$))?",
    re.IGNORECASE
)


def analyze_pages(blocks: List[Block], num_pages: int) -> Tuple[Dict[int, str], Set[int]]:
    """
    Returns:
      page_types: Dict[page_num, classification ("metadata_cover" | "answer_content" | "continuation" | "blank")]
      metadata_pages: Set of page numbers classified as cover/metadata.
    """
    blocks_by_page: Dict[int, List[Block]] = {p: [] for p in range(1, num_pages + 1)}
    for b in blocks:
        if b.page in blocks_by_page:
            blocks_by_page[b.page].append(b)

    page_types: Dict[int, str] = {}
    metadata_pages: Set[int] = set()

    for page_num in range(1, num_pages + 1):
        p_blocks = blocks_by_page[page_num]
        if not p_blocks:
            page_types[page_num] = "blank"
            continue

        full_text = " ".join(b.text.lower() for b in p_blocks)
        
        # Count metadata signals
        meta_count = sum(1 for kw in METADATA_KEYWORDS if kw in full_text)
        
        # Check for explicit cover page header
        has_cover_title = any(
            t in full_text for t in ["answer booklet", "semester examination", "student name", "roll no"]
        )

        anchor_count = sum(1 for b in p_blocks if ANCHOR_RE.match(b.text))

        if has_cover_title or (meta_count >= 2 and anchor_count == 0):
            page_types[page_num] = "metadata_cover"
            metadata_pages.add(page_num)
        elif anchor_count > 0:
            page_types[page_num] = "answer_content"
        else:
            # Check if this page is continuation of previous answer content
            if page_num > 1 and page_types.get(page_num - 1) in ("answer_content", "continuation"):
                page_types[page_num] = "continuation"
            else:
                page_types[page_num] = "answer_content"

    return page_types, metadata_pages
