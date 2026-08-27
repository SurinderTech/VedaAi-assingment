"""
Hierarchical Answer Region Segmentation & Question Anchor Detector.

Builds structured Answer Regions across lines and pages:
- Normalizes question anchors (1(a)., 1(a), Q1(a), 1.a, Q1, Q1., 1., a), b), 2(a) OR)
- Handles section context (Q1 header -> subparts a, b, c -> 1(a), 1(b), 1(c))
- Supports multi-page answer continuation (spans Page 5 and Page 6)
- Ignores cover/metadata pages flagged by page_intelligence
"""
from __future__ import annotations
import re
import uuid
from typing import List, Optional, Set, Dict
from app.models.schemas import Block, AnswerCandidate, Region, BBox

# 1. Combined anchor with parenthesis: e.g. "1(a).", "1(a)", "Q1(a)", "Ans 1(a).", "1(j)."
DIRECT_PAREN_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\s\-_]*\(([a-zA-Z0-9]{1,2})\)\s*[\.\):]?\s*(.*)$",
    re.IGNORECASE,
)

# 2. Combined anchor with dot/dash: e.g. "1.a.", "1-a", "Q1.a", "1a."
DIRECT_DOT_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\-_]\s*([a-zA-Z0-9]{1,2})\s*[\.\):]\s*(.*)$",
    re.IGNORECASE,
)

# 3. Main anchor: e.g. "Q1.", "Q1", "1.", "Ans 1", "Q7.", "Q7"
MAIN_ANCHOR_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\):]?\s*(.*)$",
    re.IGNORECASE,
)

# 4. Subpart under active main header: e.g. "a) Operating System...", "(b) Bias..."
SUBPART_ANCHOR_RE = re.compile(
    r"^\s*(?:\(([a-zA-Z0-9]{1,2})\)|([a-zA-Z0-9]{1,2})[\.\):])\s+(.*)$",
    re.IGNORECASE,
)

OR_BRANCH_RE = re.compile(
    r"^\s*(?:\[?\s*OR\s*\]?|Alternative\s+OR\s+Question\s+Reference)\s*:?\s*$",
    re.IGNORECASE,
)

HEADER_FOOTER_RE = re.compile(
    r"^\s*(?:SECTION\s*-\s*|Page\s*\d+\s*(?:of|/)\s*\d+|End\s+of\s+the\s+Question\s+Paper)",
    re.IGNORECASE,
)

GAP_MULTIPLIER = 2.2


def _clean_line_text(text: str) -> str:
    """Strips trailing mark annotations like '[2 x 5 = 10 Marks]' or '(5 Marks)'."""
    cleaned = re.sub(r"\[\s*.*?\s*Marks?\s*\]|\(\s*\d+\s*Marks?\s*\)", "", text, flags=re.IGNORECASE).strip()
    return cleaned


def _median_line_height(blocks: List[Block]) -> float:
    heights = [b.bbox.height for b in blocks if b.bbox.height > 0]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def _is_2col_mcq_page(page_blocks: List[Block]) -> bool:
    """Checks if page is a 2-column MCQ list page."""
    left_anchors = 0
    right_anchors = 0
    xs = [b.bbox.x for b in page_blocks]
    if not xs:
        return False
    mid_x = (min(xs) + max(xs)) / 2.0
    for b in page_blocks:
        txt = b.text.strip()
        if DIRECT_PAREN_RE.match(txt) or DIRECT_DOT_RE.match(txt):
            if (b.bbox.x + b.bbox.width / 2) <= mid_x:
                left_anchors += 1
            else:
                right_anchors += 1
    return left_anchors >= 3 and right_anchors >= 3


def _sort_blocks_reading_order(blocks: List[Block]) -> List[Block]:
    if not blocks:
        return []
    by_page: Dict[int, List[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    ordered: List[Block] = []
    for page_num in sorted(by_page.keys()):
        p_blocks = by_page[page_num]
        if _is_2col_mcq_page(p_blocks):
            xs = [b.bbox.x for b in p_blocks]
            mid_x = (min(xs) + max(xs)) / 2.0
            left_col = [b for b in p_blocks if (b.bbox.x + b.bbox.width / 2) <= mid_x]
            right_col = [b for b in p_blocks if (b.bbox.x + b.bbox.width / 2) > mid_x]
            left_col.sort(key=lambda b: b.bbox.y)
            right_col.sort(key=lambda b: b.bbox.y)
            ordered.extend(left_col + right_col)
        else:
            p_sorted = sorted(p_blocks, key=lambda b: b.bbox.y)
            lines: List[List[Block]] = []
            for b in p_sorted:
                placed = False
                for line in lines:
                    if abs(line[0].bbox.y - b.bbox.y) < 12:
                        line.append(b)
                        placed = True
                        break
                if not placed:
                    lines.append([b])
            for line in lines:
                line.sort(key=lambda b: b.bbox.x)
                ordered.extend(line)

    return ordered


def extract_answers(blocks: List[Block], metadata_pages: Optional[Set[int]] = None) -> List[AnswerCandidate]:
    if metadata_pages is None:
        metadata_pages = set()

    content_blocks = [b for b in blocks if b.page not in metadata_pages]
    ordered = _sort_blocks_reading_order(content_blocks)
    median_h = _median_line_height(ordered)

    candidates: List[AnswerCandidate] = []
    current: Optional[AnswerCandidate] = None

    current_main_num: Optional[str] = None
    in_or_branch: bool = False
    subpart_seen: set[str] = set()

    prev_bottom = None
    prev_page = None
    order_idx = 0

    for b in ordered:
        raw_text = b.text.strip()
        if not raw_text:
            continue

        text = _clean_line_text(raw_text) or raw_text

        # Skip section headers like "SECTION - A (COMPULSORY)" or page footers
        if HEADER_FOOTER_RE.search(text) and not (DIRECT_PAREN_RE.match(text) or DIRECT_DOT_RE.match(text)):
            continue

        if OR_BRANCH_RE.search(text):
            in_or_branch = True
            continue

        # 1. Check direct paren anchor e.g. "1(a). Deep learning..." or "1(a)."
        m_paren = DIRECT_PAREN_RE.match(text)
        if m_paren:
            main_n = m_paren.group(1)
            sub_c = m_paren.group(2).lower()
            rest = m_paren.group(3).strip()

            # Check if sub_c is digit e.g. "Q1. What is..." vs "1(a)"
            if sub_c.isdigit() and len(sub_c) == 1 and int(sub_c) > 0 and len(rest) > 10:
                q_num = main_n
                current_main_num = main_n
            else:
                current_main_num = main_n
                q_num = f"{main_n}({sub_c})"
                if in_or_branch:
                    q_num = f"{main_n}({sub_c}) OR"

            current = AnswerCandidate(
                answer_id=f"answer_{uuid.uuid4().hex[:8]}",
                question_number=q_num,
                text=raw_text,
                regions=[Region(page=b.page, bbox=b.bbox)],
                order_index=order_idx,
            )
            order_idx += 1
            candidates.append(current)
            prev_bottom = b.bbox.y + b.bbox.height
            prev_page = b.page
            continue

        # 2. Check direct dot anchor e.g. "1.a. Deep learning..."
        m_dot = DIRECT_DOT_RE.match(text)
        if m_dot:
            main_n = m_dot.group(1)
            sub_c = m_dot.group(2).lower()
            rest = m_dot.group(3).strip()

            current_main_num = main_n
            q_num = f"{main_n}({sub_c})"
            if in_or_branch:
                q_num = f"{main_n}({sub_c}) OR"

            current = AnswerCandidate(
                answer_id=f"answer_{uuid.uuid4().hex[:8]}",
                question_number=q_num,
                text=raw_text,
                regions=[Region(page=b.page, bbox=b.bbox)],
                order_index=order_idx,
            )
            order_idx += 1
            candidates.append(current)
            prev_bottom = b.bbox.y + b.bbox.height
            prev_page = b.page
            continue

        # 3. Check subpart anchor under active main header e.g. "a) Operating System..."
        m_sub = SUBPART_ANCHOR_RE.match(text)
        if m_sub and current_main_num is not None:
            sub_char = (m_sub.group(1) or m_sub.group(2)).lower()
            rest_text = m_sub.group(3).strip()

            q_num = f"{current_main_num}({sub_char})"
            if in_or_branch or sub_char in subpart_seen:
                q_num = f"{current_main_num}({sub_char}) OR"

            current = AnswerCandidate(
                answer_id=f"answer_{uuid.uuid4().hex[:8]}",
                question_number=q_num,
                text=rest_text or raw_text,
                regions=[Region(page=b.page, bbox=b.bbox)],
                order_index=order_idx,
            )
            order_idx += 1
            candidates.append(current)
            subpart_seen.add(sub_char)
            prev_bottom = b.bbox.y + b.bbox.height
            prev_page = b.page
            continue

        # 4. Check Main Anchor e.g. "Q2. Operating Systems", "Q7. Explain..."
        m_main = MAIN_ANCHOR_RE.match(text)
        if m_main and (text.lower().startswith("q") or text.lower().startswith("ans") or len(text) < 70):
            main_num = m_main.group(1)
            current_main_num = main_num
            in_or_branch = False
            subpart_seen = set()

            current = AnswerCandidate(
                answer_id=f"answer_{uuid.uuid4().hex[:8]}",
                question_number=main_num,
                text=raw_text,
                regions=[Region(page=b.page, bbox=b.bbox)],
                order_index=order_idx,
            )
            order_idx += 1
            candidates.append(current)
            prev_bottom = b.bbox.y + b.bbox.height
            prev_page = b.page
            continue

        # 5. Gap detection & continuation across lines/pages
        starts_new_by_gap = (
            prev_page == b.page
            and prev_bottom is not None
            and (b.bbox.y - prev_bottom) > median_h * GAP_MULTIPLIER
        )

        if current is None or (starts_new_by_gap and not current.text):
            current = AnswerCandidate(
                answer_id=f"answer_{uuid.uuid4().hex[:8]}",
                question_number=None,
                text=text,
                regions=[Region(page=b.page, bbox=b.bbox)],
                order_index=order_idx,
            )
            order_idx += 1
            candidates.append(current)
        else:
            # Continuation line of active answer candidate
            current.text = f"{current.text} {text}".strip()
            current.regions.append(Region(page=b.page, bbox=b.bbox))

        prev_bottom = b.bbox.y + b.bbox.height
        prev_page = b.page

    return candidates

