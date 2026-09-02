"""
Page Intelligence — Deprecated Legacy Stub.
Page understanding is now handled natively by Multimodal VLM visual intelligence.
"""
from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from app.models.schemas import Block, PageAnalysis, QuestionAnchor


def analyze_pages(
    blocks: List[Block],
    num_pages: int,
    anchors_by_page: Optional[Dict[int, List[QuestionAnchor]]] = None,
) -> Tuple[Dict[int, PageAnalysis], Set[int]]:
    """Deprecated stub: Multimodal VLM directly understands page roles visually."""
    analyses: Dict[int, PageAnalysis] = {
        p: PageAnalysis(
            page=p,
            classification="ANSWER_CONTENT",
            confidence=1.0,
            metadata_likelihood=0.0,
            ocr_density=1.0,
            anchors=[],
        )
        for p in range(1, num_pages + 1)
    }
    return analyses, set()
