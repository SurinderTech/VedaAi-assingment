"""
Answer extraction (plan sections 13-15).

The answer sheet is handwritten, so OCR confidence is inherently lower.
We segment lines into answer candidates using two signals:
  1. An explicit question-number marker the student wrote (same regex
     family as question_extractor, reused loosely).
  2. Vertical gap detection: a large jump in y between consecutive lines
     signals a new answer block even with no number written.

Each candidate keeps ALL of its line regions (not just one bbox), so an
answer can span multiple visual blocks and multiple pages (section 25).
"""
from __future__ import annotations
import re
import uuid
from typing import List
from app.models.schemas import Block, AnswerCandidate, Region, BBox

# Looser than the question-paper regex: students write "11(a)", "Ans 3",
# "3)", "Q3", etc. in inconsistent handwriting-OCR'd form.
NUMBER_MARKER_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*)?(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*(\([a-zA-Z]\))?\s*[\.\):]?\s*(.*)$"
)

GAP_MULTIPLIER = 2.2  # a gap this many times the median line height = new block


def _median_line_height(blocks: List[Block]) -> float:
    heights = [b.bbox.height for b in blocks if b.bbox.height > 0]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def extract_answers(blocks: List[Block]) -> List[AnswerCandidate]:
    ordered = sorted(blocks, key=lambda b: (b.page, b.bbox.y))
    median_h = _median_line_height(ordered)

    candidates: List[AnswerCandidate] = []
    current: AnswerCandidate | None = None
    prev_bottom = None
    prev_page = None
    order_index = 0

    for b in ordered:
        text = b.text.strip()
        if not text:
            continue

        m = NUMBER_MARKER_RE.match(text)
        starts_new_by_number = bool(m and m.group(1))
        starts_new_by_gap = (
            prev_page == b.page
            and prev_bottom is not None
            and (b.bbox.y - prev_bottom) > median_h * GAP_MULTIPLIER
        )
        starts_new_by_page = prev_page is not None and b.page != prev_page and current is None

        if current is None or starts_new_by_number or (starts_new_by_gap and current is not None):
            q_number = None
            body = text
            if starts_new_by_number:
                main, sub, rest = m.group(1), m.group(2), m.group(3)
                sub_clean = sub.strip("() ") if sub else None
                q_number = f"{main}({sub_clean})" if sub_clean else main
                body = rest.strip() or text

            current = AnswerCandidate(
                answer_id=f"answer_{uuid.uuid4().hex[:8]}",
                question_number=q_number,
                text=body,
                regions=[Region(page=b.page, bbox=b.bbox)],
                order_index=order_index,
            )
            order_index += 1
            candidates.append(current)
        else:
            current.text = f"{current.text} {text}".strip()
            current.regions.append(Region(page=b.page, bbox=b.bbox))

        prev_bottom = b.bbox.y + b.bbox.height
        prev_page = b.page

    return candidates
