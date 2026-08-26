"""
Question extraction engine (plan sections 11-12).

Handles question markers whether on the same line or standalone line,
including subquestions like 11(a), 11(b), Q1, Question 1, etc.
Preserves original numbering and order.
"""
from __future__ import annotations
import re
from typing import List
from app.models.schemas import Block, Question

# Matches question markers with or without text on the same line
# Examples: "11(a) Explain...", "11(a)", "11 (a).", "Q1.", "Question 1:", "1."
QUESTION_MARKER_RE = re.compile(
    r"""^\s*
    (?:Q(?:uestion)?\.?\s*)?                   # Optional "Q", "Q.", "Question" prefix
    (\d{1,3})                                   # Main number (e.g. 1, 11)
    \s*
    (\([a-zA-Z]\)|[a-zA-Z](?=[\s.):]|$))?      # Subpart (e.g. (a), a, (b), b)
    \s*
    [\.\):]?                                    # Trailing punctuation
    (?:\s+(.*)|\s*$)                           # Question text on same line (or empty)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Standalone subpart marker if question number was separate (e.g. Q11 on line 1, (a) on line 2)
STANDALONE_SUBPART_RE = re.compile(
    r"^\s*(?:\(([a-zA-Z])\)|([a-zA-Z])[\.\):])\s*(.*)$"
)


def _normalize_number(main: str, sub: str | None) -> str:
    if not sub:
        return main
    clean_sub = sub.strip("() ").lower()
    return f"{main}({clean_sub})"


def extract_questions(blocks: List[Block]) -> List[Question]:
    # Sort page by page, top to bottom
    ordered = sorted(blocks, key=lambda b: (b.page, b.bbox.y))

    questions: List[Question] = []
    current: Question | None = None
    order_index = 0

    for b in ordered:
        text = b.text.strip()
        if not text:
            continue

        m = QUESTION_MARKER_RE.match(text)
        if m:
            main, sub, rest = m.group(1), m.group(2), m.group(3)
            number = _normalize_number(main, sub)
            initial_text = (rest or "").strip()

            current = Question(
                id=number,
                number=number,
                text=initial_text,
                page=b.page,
                bbox=b.bbox,
                order_index=order_index,
            )
            order_index += 1
            questions.append(current)
            continue

        # Check for standalone subpart if current question exists and has main number
        sub_match = STANDALONE_SUBPART_RE.match(text)
        if sub_match and current is not None:
            sub_char = sub_match.group(1) or sub_match.group(2)
            rest_text = (sub_match.group(3) or "").strip()
            # Extract main number from current question
            main_num = re.split(r"\(|\.", current.number)[0]
            number = f"{main_num}({sub_char.lower()})"

            # If current question had no text, transform it into this subquestion
            if not current.text:
                current.id = number
                current.number = number
                current.text = rest_text
            else:
                current = Question(
                    id=number,
                    number=number,
                    text=rest_text,
                    page=b.page,
                    bbox=b.bbox,
                    order_index=order_index,
                )
                order_index += 1
                questions.append(current)
            continue

        # Continuation text of the current question
        if current is not None:
            if current.text:
                current.text = f"{current.text} {text}".strip()
            else:
                current.text = text

    # Deduplicate questions by ID defensively while preserving order
    seen = set()
    unique: List[Question] = []
    for q in questions:
        if q.id in seen:
            continue
        seen.add(q.id)
        unique.append(q)

    return unique
