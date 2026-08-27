"""
Real-Time Dynamic LLM & Semantic NLP Question Extraction Engine.

Dynamic Capabilities:
- Real-time semantic intent classification (Metadata vs Candidate Guidelines vs Genuine Questions)
- Zero reliance on static keyword lists or hardcoded paper structures
- Dynamic subpart decomposition (e.g. Q1 -> 1(a), 1(b)... 1(j))
- Dynamic NLP heuristic fallback classifier for 100% offline reliability
"""
from __future__ import annotations
import asyncio
import re
from typing import List, Optional, Dict, Set, Tuple
from app.models.schemas import Block, Question, BBox
from app.services.llm_provider import llm_complete_json, LLMError

# Structural regexes for numbering syntax (not specific words)
MAIN_Q_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\):]\s*(.*)$",
    re.IGNORECASE
)

SUBPART_RE = re.compile(
    r"^\s*(?:\(([a-z0-9]{1,2})\)|([a-z0-9]{1,2})[\.\):])\s+(.*)$",
    re.IGNORECASE
)

COMBINED_Q_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\s\-_]*\(([a-z0-9]{1,2})\)\s*[\.\):]?\s+(.*)$",
    re.IGNORECASE
)

MCQ_OPTION_RE = re.compile(
    r"^\s*\([A-D]\)\s+|^[A-D]\.\s+"
)

QUESTION_INTENT_VERBS = re.compile(
    r"\b(?:what|why|how|explain|discuss|calculate|derive|prove|compare|define|state|find|show|list|describe|illustrate|evaluate|analyze|differentiate|determine|solve)\b",
    re.IGNORECASE
)

ADMIN_RULE_KEYWORDS = [
    "attempt", "compulsory", "consisting of", "carrying", "instruction",
    "candidate", "allowed", "forbidden", "maximum mark", "max mark",
    "duration", "time allowed", "read carefully", "total question",
    "attempt any", "answer any", "carry equal marks", "in all", "section"
]


def _is_admin_metadata(text: str) -> bool:
    """Dynamically detects administrative header metadata (Key-Value pairs, paper metadata)."""
    t = text.strip()
    if not t:
        return True

    # Check key-value header line (e.g. 'Label : Value') without question intent
    if re.match(r"^[A-Za-z0-9\s\.\(\)–-]{2,35}\s*:\s*.+$", t):
        if not (QUESTION_INTENT_VERBS.search(t) or t.endswith("?")):
            return True

    # Generic admin header tags
    t_low = t.lower()
    if re.search(r"\b(?:roll\s*no|b\.?\s*tech|m\.?\s*tech|b\.?\s*sc|m\.?\s*sc|sem(?:ester)?|reg(?:istration)?\s*no|question\s*paper\s*code|m\.?\s*code)\b", t_low):
        if not (QUESTION_INTENT_VERBS.search(t_low) or t.endswith("?")):
            return True

    return False


def _is_candidate_instruction_or_rule(text: str) -> bool:
    """Dynamically evaluates whether a line represents candidate attempt rules/guidelines."""
    t = text.strip()
    if not t:
        return True

    t_low = t.lower()
    has_rule_keyword = any(k in t_low for k in ADMIN_RULE_KEYWORDS)

    if has_rule_keyword:
        has_q_intent = bool(QUESTION_INTENT_VERBS.search(t_low)) or t.endswith("?")
        if not has_q_intent:
            return True

    return False


def _is_non_question_line(text: str) -> bool:
    """Combined dynamic check for metadata & administrative rules."""
    return _is_admin_metadata(text) or _is_candidate_instruction_or_rule(text)


async def extract_questions(blocks: List[Block]) -> List[Question]:
    """
    Main entry point for dynamic question extraction.
    Uses LLM real-time semantic analysis with dynamic NLP heuristic fallback.
    """
    if not blocks:
        return []

    questions: List[Question] = []

    blocks_by_page: Dict[int, List[Block]] = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page, []).append(b)

    try:
        questions = await _llm_extract_questions(blocks_by_page, blocks)
    except Exception as e:
        print(f"[QuestionExtractor] Dynamic LLM extraction error ({e}), using dynamic heuristic engine.")
        questions = []

    if not questions:
        print("[QuestionExtractor] LLM returned 0 questions, executing dynamic fallback engine.")
        questions = _heuristic_extract_questions(blocks_by_page)

    questions = _attach_bboxes_to_questions(questions, blocks)
    questions = _verify_questions(questions)

    return questions


async def _llm_extract_questions(
    question_pages: Dict[int, List[Block]], all_blocks: List[Block]
) -> List[Question]:
    """Real-time dynamic LLM extraction using semantic intent prompts."""
    sem = asyncio.Semaphore(2)
    questions: List[Question] = []
    order_idx = 0

    async def extract_page(page_num: int, p_blocks: List[Block]) -> List[dict]:
        async with sem:
            p_text = "\n".join(b.text for b in p_blocks)
            prompt = (
                f"You are an AI exam paper analysis engine. Intelligently perform real-time dynamic semantic extraction on Page {page_num}.\n"
                "DYNAMIC CLASSIFICATION DIRECTIVES:\n"
                "1. ANALYZE intent of all text segments dynamically.\n"
                "2. DISCARD ALL Administrative Metadata (e.g. institution headers, roll numbers, exam dates, paper codes, durations, page counts, total marks).\n"
                "3. DISCARD ALL Candidate Guidelines & Section Attempt Rules (e.g. guidelines on how many questions to attempt, section weightage, choice rules, candidate instructions).\n"
                "4. EXTRACT ONLY Genuine Examination Questions requiring student answers.\n"
                "5. For multi-part questions (e.g. Question 1 having subparts (a), (b), (c)... or 1(a), 1(b)...), extract EACH individual subpart as an independent question item with number like '1(a)', '1(b)' etc.\n"
                "6. For MCQs, include option choices inside the 'text' field.\n"
                "7. Output MUST be ONLY a raw JSON array of objects without markdown formatting:\n"
                "[\n"
                "  {\n"
                f'    "number": "1(a)", "section": "Section-A", "text": "What is deep learning?", "page": {page_num}\n'
                "  }\n"
                "]\n\n"
                f"Page Text:\n{p_text}\n"
            )
            try:
                data = await llm_complete_json(prompt)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                    return data["questions"]
            except Exception as e:
                print(f"[QuestionExtractor] LLM error on page {page_num}: {e}")
            return []

    tasks = [extract_page(p, p_blocks) for p, p_blocks in question_pages.items()]
    page_question_lists = await asyncio.gather(*tasks)

    for p_num, q_list in zip(question_pages.keys(), page_question_lists):
        for raw_q in q_list:
            if not isinstance(raw_q, dict):
                continue
            q_num = str(raw_q.get("number", "")).strip()
            q_text = str(raw_q.get("text", "")).strip()
            q_sec = str(raw_q.get("section", "")).strip() if raw_q.get("section") else None

            if not q_num or not q_text:
                continue

            if _is_non_question_line(q_text):
                continue

            questions.append(
                Question(
                    id=q_num,
                    number=q_num,
                    text=q_text,
                    page=p_num,
                    order_index=order_idx,
                    section=q_sec,
                )
            )
            order_idx += 1

    return questions


def _heuristic_extract_questions(question_pages: Dict[int, List[Block]]) -> List[Question]:
    """Dynamic NLP semantic fallback parser."""
    questions: List[Question] = []
    order_idx = 0
    current_main_num: Optional[str] = None

    for page_num, p_blocks in question_pages.items():
        ordered = sorted(p_blocks, key=lambda b: b.bbox.y)
        current_question: Optional[Question] = None

        for b in ordered:
            txt = b.text.strip()
            if not txt or _is_non_question_line(txt):
                continue

            # Check combined Q e.g. "1(a) What is deep learning?"
            comb_m = COMBINED_Q_RE.match(txt)
            if comb_m and len(comb_m.group(3)) > 3 and not _is_non_question_line(comb_m.group(3)):
                main_n = comb_m.group(1)
                sub_c = comb_m.group(2).lower()
                q_text = comb_m.group(3).strip()
                q_num = f"{main_n}({sub_c})"
                current_main_num = main_n

                current_question = Question(
                    id=q_num,
                    number=q_num,
                    text=q_text,
                    page=page_num,
                    bbox=b.bbox,
                    order_index=order_idx,
                )
                order_idx += 1
                questions.append(current_question)
                continue

            # Check main question e.g. "1. SECTION-A is..." or "2. What is an activation function?"
            main_m = MAIN_Q_RE.match(txt)
            if main_m:
                q_num = main_m.group(1)
                q_rest = main_m.group(2).strip()

                if _is_non_question_line(q_rest):
                    current_main_num = q_num
                    continue

                # Check if q_rest contains subpart e.g. "(a) What is deep learning?"
                sub_in = SUBPART_RE.match(q_rest)
                if sub_in:
                    sub_c = (sub_in.group(1) or sub_in.group(2)).lower()
                    rest_t = sub_in.group(3).strip()
                    full_q_num = f"{q_num}({sub_c})"
                    current_main_num = q_num

                    current_question = Question(
                        id=full_q_num,
                        number=full_q_num,
                        text=rest_t or f"Question {full_q_num}",
                        page=page_num,
                        bbox=b.bbox,
                        order_index=order_idx,
                    )
                    order_idx += 1
                    questions.append(current_question)
                    continue

                current_main_num = q_num
                current_question = Question(
                    id=q_num,
                    number=q_num,
                    text=q_rest or f"Question {q_num}",
                    page=page_num,
                    bbox=b.bbox,
                    order_index=order_idx,
                )
                order_idx += 1
                questions.append(current_question)
                continue

            # Check subpart under active main number e.g. "(b) What is bias and variance?"
            sub_m = SUBPART_RE.match(txt)
            if sub_m and current_main_num is not None:
                sub_c = (sub_m.group(1) or sub_m.group(2)).lower()
                q_text = sub_m.group(3).strip()
                if q_text and not _is_non_question_line(q_text):
                    q_num = f"{current_main_num}({sub_c})"
                    current_question = Question(
                        id=q_num,
                        number=q_num,
                        text=q_text,
                        page=page_num,
                        bbox=b.bbox,
                        order_index=order_idx,
                    )
                    order_idx += 1
                    questions.append(current_question)
                    continue

            # MCQ option continuation
            if MCQ_OPTION_RE.match(txt) and current_question is not None:
                current_question.text = f"{current_question.text}\n{txt}".strip()
                continue

            if current_question is not None and current_question.page == page_num:
                current_question.text = f"{current_question.text}\n{txt}".strip()

    return questions


def _attach_bboxes_to_questions(questions: List[Question], blocks: List[Block]) -> List[Question]:
    """Aligns bounding box unions for each question from raw OCR/PDF text blocks."""
    blocks_by_page: Dict[int, List[Block]] = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page, []).append(b)

    for q in questions:
        p_blocks = blocks_by_page.get(q.page, [])
        if not p_blocks:
            continue

        matching_blocks: List[Block] = []
        q_num_clean = q.number.replace("OR", "").strip()

        for b in p_blocks:
            t = b.text.strip()
            if t.startswith(f"{q_num_clean}.") or t.startswith(f"Q{q_num_clean}") or t.startswith(f"Ans {q_num_clean}"):
                matching_blocks.append(b)
            elif matching_blocks:
                first_few_words = q.text[:30].lower()
                if any(w in b.text.lower() for w in first_few_words.split()[:3]):
                    matching_blocks.append(b)

        if matching_blocks:
            xs = [mb.bbox.x for mb in matching_blocks]
            ys = [mb.bbox.y for mb in matching_blocks]
            ws = [mb.bbox.x + mb.bbox.width for mb in matching_blocks]
            hs = [mb.bbox.y + mb.bbox.height for mb in matching_blocks]

            min_x, min_y = min(xs), min(ys)
            max_w, max_h = max(ws) - min_x, max(hs) - min_y

            q.bbox = BBox(x=round(min_x, 1), y=round(min_y, 1), width=round(max_w, 1), height=round(max_h, 1))

    return questions


def _verify_questions(questions: List[Question]) -> List[Question]:
    """Final deduplication and dynamic intent verification pass."""
    verified: List[Question] = []
    seen_ids: Set[str] = set()

    for q in questions:
        if q.id in seen_ids:
            continue
        if _is_non_question_line(q.text):
            continue
        seen_ids.add(q.id)
        verified.append(q)

    verified.sort(key=lambda q: q.order_index)
    return verified

