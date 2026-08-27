"""
Question extraction engine.

Handles:
- Main question headers (e.g., "Q1. Answer the following...", "Q2. Operating Systems (10 Marks)")
- Subquestions (e.g., "a) Define an operating system...", "b) What is a thread...")
- OR branch handling (e.g., "OR", "a) Explain deadlock conditions...") -> 2(a) OR
- Noise block filtering (e.g. ASCII diagram lines like "-", "---", "| |")

Ensures ONLY actual subquestions / evaluatable questions are returned.
"""
from __future__ import annotations
import re
from typing import List, Optional
from app.models.schemas import Block, Question

MAIN_Q_RE = re.compile(
    r"^\s*Q(?:uestion)?\.?\s*(\d{1,3})\s*[\.\):]?\s*(.*)$", re.IGNORECASE
)
MAIN_NUM_ONLY_RE = re.compile(
    r"^\s*(\d{1,3})\s*[\.\)]\s*(.*)$"
)

SUBPART_RE = re.compile(
    r"^\s*(?:\(([a-zA-Z1-9])\)|([a-zA-Z])[\.\):])\s*(.*)$"
)

NOISE_RE = re.compile(
    r"^[\s\-_=\|\\\/\+\*\.]{1,30}$"
)

OR_RE = re.compile(r"^\s*OR\s*$", re.IGNORECASE)


def is_noise(text: str) -> bool:
    t = text.strip()
    if not t or (len(t) < 2 and not t.isalnum()):
        return True
    if NOISE_RE.match(t):
        return True
    return False


def extract_questions(blocks: List[Block]) -> List[Question]:
    ordered = sorted(blocks, key=lambda b: (b.page, b.bbox.y))
    questions: List[Question] = []

    current_main_num: Optional[str] = None
    current_main_header: str = ""
    in_or_branch: bool = False
    subpart_seen: set[str] = set()

    current_question: Optional[Question] = None
    order_idx = 0

    for b in ordered:
        text = b.text.strip()
        if not text or is_noise(text):
            continue

        # Check for OR branch marker
        if OR_RE.match(text):
            in_or_branch = True
            continue

        # Check for Main Question Header: "Q1. Answer the following...", "Q2. Operating Systems..."
        main_m = MAIN_Q_RE.match(text)
        if not main_m:
            num_m = MAIN_NUM_ONLY_RE.match(text)
            if num_m and int(num_m.group(1)) <= 30 and not current_question:
                main_m = num_m

        if main_m:
            current_main_num = main_m.group(1)
            current_main_header = main_m.group(2).strip()
            in_or_branch = False
            subpart_seen = set()

            rest = current_main_header
            sub_m = SUBPART_RE.match(rest)
            if sub_m:
                sub_char = (sub_m.group(1) or sub_m.group(2)).lower()
                q_num = f"{current_main_num}({sub_char})"
                q_text = sub_m.group(3).strip()
                current_question = Question(
                    id=q_num, number=q_num, text=q_text, page=b.page, bbox=b.bbox, order_index=order_idx
                )
                order_idx += 1
                questions.append(current_question)
                subpart_seen.add(sub_char)
            else:
                is_header_only = any(
                    k in rest.lower() for k in ["answer the following", "marks", "consider", "section", "part"]
                ) or len(rest) < 12

                if not is_header_only and rest:
                    current_question = Question(
                        id=current_main_num, number=current_main_num, text=rest, page=b.page, bbox=b.bbox, order_index=order_idx
                    )
                    order_idx += 1
                    questions.append(current_question)
                else:
                    current_question = None
            continue

        # Check for subpart line e.g. "a) Define an operating system..."
        sub_m = SUBPART_RE.match(text)
        if sub_m and current_main_num is not None:
            sub_char = (sub_m.group(1) or sub_m.group(2)).lower()
            rest_text = sub_m.group(3).strip()

            q_num = f"{current_main_num}({sub_char})"
            if in_or_branch or sub_char in subpart_seen:
                q_num = f"{current_main_num}({sub_char}) OR"

            current_question = Question(
                id=q_num,
                number=q_num,
                text=rest_text,
                page=b.page,
                bbox=b.bbox,
                order_index=order_idx,
            )
            order_idx += 1
            questions.append(current_question)
            subpart_seen.add(sub_char)
            continue

        # Continuation line of current question
        if current_question is not None:
            if current_question.text:
                current_question.text = f"{current_question.text} {text}".strip()
            else:
                current_question.text = text

    # Deduplicate questions while preserving order
    seen = set()
    unique: List[Question] = []
    for q in questions:
        if q.id in seen:
            continue
        seen.add(q.id)
        unique.append(q)

    return unique

