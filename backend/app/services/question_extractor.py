"""
Layered Question Paper Extraction Pipeline.

Configurable & Multi-Signal Architecture:
- Thresholds are fully configurable via settings (app.core.config) and function arguments.
- Intent verbs and keyword lists serve strictly as weighted supporting signals,
  never required to qualify as a question, and never the sole reason for discarding text.

Pipeline Stages:
PDF/image -> text + layout/coordinates -> multi-signal candidate scoring
-> parent/subquestion hierarchy resolution -> multi-line continuation grouping
-> configurable threshold classification -> targeted LLM fallback (ambiguous candidates only)
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Dict, Set, Tuple
from app.models.schemas import Block, Question, BBox
from app.core.config import settings
from app.services.llm_provider import llm_complete_json

# --- Regex Patterns for Question Numbering Syntax ---

# Combined main + subquestion: e.g. 1(a), 1. (a), Q1(a), 11(a), 11-a, Q.1(a)
COMBINED_Q_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})\s*[\.\:\-\s]*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)$",
    re.IGNORECASE
)

# Main question marker: e.g. 1., 1), Q1, Q1., Q.1, Q 1, Question 1:, Q 1.
MAIN_Q_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})\s*[\.\):\-]\s*(.*)$",
    re.IGNORECASE
)

# Standalone Q / Q. prefix followed by digit and space: e.g. Q1 What is..., Q.1 Discuss...
MAIN_Q_PREFIX_RE = re.compile(
    r"^\s*Q\.?\s*(\d{1,3})\s+(.*)$",
    re.IGNORECASE
)

# Independent subquestion under active group parent: e.g. (a) ..., a) ..., a. ..., [a] ...
INDEPENDENT_SUB_RE = re.compile(
    r"^\s*[\(\[]?\s*([a-z]{1,2}|[ivxlcdm]{1,4})\s*[\)\]\.\:\-]\s+(.*)$",
    re.IGNORECASE
)

# Valid Subquestion Letter Matcher (Pure letters/roman numerals without embedded digits like S2)
VALID_SUBPART_LETTER = re.compile(r"^[a-z]{1,2}$|^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)$", re.IGNORECASE)

# Supporting Intent Verbs & Action Markers (Supporting signal only; NOT required)
SUPPORTING_INTENT_VERBS = re.compile(
    r"\b(?:what|why|how|explain|discuss|calculate|derive|prove|compare|define|state|find|show|list|describe|illustrate|evaluate|analyze|differentiate|determine|solve|convert|identify|construct|write|compute|trace|sketch|distinguish|design|briefly)\b",
    re.IGNORECASE
)

# Header Metadata Terms (Used in multi-signal context scoring, not sole decider)
METADATA_HEADER_PATTERNS = [
    r"\broll\s*no\b",
    r"\btotal\s*no\.\s*of\s*pages\b",
    r"\btotal\s*no\.\s*of\s*questions\b",
    r"\bb\.?\s*tech\b",
    r"\bm\.?\s*tech\b",
    r"\bb\.?\s*sc\b",
    r"\bm\.?\s*sc\b",
    r"\bsem(?:ester)?\b",
    r"\bsubject\s*code\b",
    r"\bm\.?\s*code\b",
    r"\bdate\s*of\s*examination\b",
    r"\btime\s*:\s*\d+",
    r"\bmax\.?\s*marks\b",
    r"\bmaximum\s*marks\b",
    r"\btime\s*allowed\b",
    r"\breg(?:istration)?\s*no\b",
    r"\bbranch\b",
    r"\bpaper\s*code\b",
    r"\bm\-\s*\([a-z0-9]+\)\-\d+",
    r"\bm\-\s*\d+",
]

# Section Headers
SECTION_HEADER_RE = re.compile(
    r"^\s*(?:SECTION|PART|GROUP)\s*[\-\–\:\s]*([A-Z0-9]{1,3})\s*$",
    re.IGNORECASE
)

# Administrative & Candidate Attempt Rule Terms (Supporting signal in multi-signal scoring)
ADMIN_RULE_KEYWORDS = [
    "attempt", "compulsory", "consisting of", "carrying", "instruction",
    "candidate", "allowed", "forbidden", "maximum mark", "max mark",
    "duration", "time allowed", "read carefully", "total question",
    "attempt any", "answer any", "carry equal marks", "in all"
]

# Common OCR typo fixes for line continuation
OCR_HYPHEN_FIXES = [
    (r"\bsequence-to-s\s+equence\b", "sequence-to-sequence"),
    (r"\bcomputat\s+ional\b", "computational"),
    (r"\bneuroscie\s+ntific\b", "neuroscientific"),
    (r"\bnetwor\s+ks\b", "networks"),
    (r"\bdiff\s+erences\b", "differences"),
    (r"\bvario\s+us\b", "various"),
]


def _clean_ocr_artifacts(text: str) -> str:
    """Fixes known OCR hyphenation splits mid-word."""
    cleaned = text
    for pattern, replacement in OCR_HYPHEN_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _is_watermark_or_noise(text: str) -> bool:
    """Identifies document watermarks, domain footers, or page pagination lines."""
    t = text.strip().lower()
    if not t:
        return True
    if "brpaper.com" in t or "aglasem.com" in t:
        return True
    if re.match(r"^\d{1,2}\s*\|\s*(?:m\-)?.*$", t):
        return True
    return False


def _is_section_header(text: str) -> Tuple[bool, Optional[str]]:
    """Detects standalone section headers like 'SECTION-A' or 'PART-B'."""
    t = text.strip()
    m = SECTION_HEADER_RE.match(t)
    if m:
        sec_name = f"Section-{m.group(1).upper()}"
        return True, sec_name
    return False, None


def _calculate_candidate_confidence(
    text: str,
    number_pattern_matched: bool,
    is_subquestion: bool,
    page_num: int,
    y_ratio: float = 0.5,
) -> float:
    """
    Weighted Multi-Signal Scoring Engine.
    
    Signals & Weights:
    - Structural Numbering Match (+0.50 to +0.65)
    - Subquestion Sequence (+0.60)
    - Intent Verb present (+0.25, SUPPORTING ONLY, not required)
    - Question Mark ? or Math/Code syntax (+0.25)
    - Exam Attempt Rule (e.g. "SECTION-B contains 5 questions carrying 5 marks each...") (-0.65)
    - Top of Page Header Key-Value Metadata without Question Marker (-0.70)
    
    No single keyword is the sole decision maker, and intent verbs are never mandatory.
    """
    t = text.strip()
    if not t or _is_watermark_or_noise(t):
        return 0.0

    t_low = t.lower()
    score = 0.0

    # 1. Structural Numbering Signal (Strong Positive)
    if is_subquestion:
        score += 0.65
    elif number_pattern_matched:
        score += 0.50

    # 2. Supporting Question Syntax Signals (Positive, optional)
    has_q_intent = False
    if t.endswith("?") or "?" in t:
        score += 0.25
        has_q_intent = True

    if SUPPORTING_INTENT_VERBS.search(t_low):
        score += 0.20
        has_q_intent = True

    # Math / Equation / Marks syntax support (Positive)
    if re.search(r"[\=\+\-\*\/\$\\]|\[\d+\s*marks?\]|\(\d+\)", t_low):
        score += 0.10

    # 3. Instruction & Attempt Rule Penalty (Evaluated if describing exam rules without question intent)
    if "compulsory" in t_low or "consisting of" in t_low or "carrying" in t_low or "attempt any" in t_low or "have to attempt" in t_low:
        if ("section" in t_low or "part" in t_low or "questions" in t_low) and not has_q_intent:
            score -= 0.65

    if "instructions to candidates" in t_low or "general instructions" in t_low:
        score -= 0.80

    # 4. Top-of-Page Key-Value Metadata Penalty (Only if line DOES NOT have a question marker)
    if not number_pattern_matched and not is_subquestion:
        is_kv_header = bool(re.match(r"^[A-Za-z0-9\s\.\(\)\–\-]{2,35}\s*:\s*.+$", t))
        has_meta_kw = any(re.search(pat, t_low) for pat in METADATA_HEADER_PATTERNS)

        if (is_kv_header or has_meta_kw) and page_num == 1 and y_ratio < 0.35:
            score -= 0.70

    # Clamp confidence to [0.0, 1.0]
    return max(0.0, min(1.0, round(score, 3)))


def _is_group_parent_header(text: str) -> bool:
    """
    Detects group parent headers like '1. Write briefly :' or '2. Operating Systems (10 Marks)'
    which introduce subquestions rather than forming a standalone question.
    """
    t = text.strip()
    m = MAIN_Q_RE.match(t) or MAIN_Q_PREFIX_RE.match(t)
    if m:
        body = m.group(2).strip()
        body_low = body.lower()

        if t.endswith(":") or "write briefly" in body_low or "answer the following" in body_low or "short answer questions" in body_low:
            return True

        if re.search(r"\(\d{1,2}\s*marks?\)", body_low) or body_low.endswith("marks)") or body_low.endswith("marks"):
            if not (SUPPORTING_INTENT_VERBS.search(body_low) or body.endswith("?")):
                return True

        if len(body) < 25 and not (SUPPORTING_INTENT_VERBS.search(body_low) or body.endswith("?")):
            return True

    return False


class RawCandidate:
    def __init__(
        self,
        q_num: str,
        display_num: str,
        text: str,
        page: int,
        blocks: List[Block],
        section: Optional[str] = None,
        confidence: float = 0.9,
    ):
        self.q_num = q_num
        self.display_num = display_num
        self.text = text
        self.page = page
        self.blocks = blocks
        self.section = section
        self.confidence = confidence

    @property
    def bbox(self) -> Optional[BBox]:
        if not self.blocks:
            return None
        xs = [b.bbox.x for b in self.blocks]
        ys = [b.bbox.y for b in self.blocks]
        ws = [b.bbox.x + b.bbox.width for b in self.blocks]
        hs = [b.bbox.y + b.bbox.height for b in self.blocks]
        min_x, min_y = min(xs), min(ys)
        max_w, max_h = max(ws) - min_x, max(hs) - min_y
        return BBox(x=round(min_x, 1), y=round(min_y, 1), width=round(max_w, 1), height=round(max_h, 1))


def _extract_candidates_from_blocks(blocks: List[Block]) -> List[RawCandidate]:
    """
    Layered local structural extraction pass.
    Identifies questions, subquestions, continuations, and section contexts.
    """
    candidates: List[RawCandidate] = []

    blocks_by_page: Dict[int, List[Block]] = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page, []).append(b)

    active_section: Optional[str] = None
    current_main_num: Optional[str] = None

    for page_num in sorted(blocks_by_page.keys()):
        p_blocks = sorted(blocks_by_page[page_num], key=lambda b: (b.bbox.y, b.bbox.x))
        curr_candidate: Optional[RawCandidate] = None

        for b in p_blocks:
            txt = b.text.strip()
            if not txt or _is_watermark_or_noise(txt):
                continue

            # 1. Section Header check
            is_sec, sec_name = _is_section_header(txt)
            if is_sec:
                active_section = sec_name
                curr_candidate = None
                continue

            # 2. Check Combined Subquestion: e.g. "1(a) Define deep learning." or "11 (a) Solve x^2..."
            comb_m = COMBINED_Q_RE.match(txt)
            if comb_m:
                main_n = comb_m.group(1)
                sub_c = comb_m.group(2).lower()
                body = comb_m.group(3).strip()

                if VALID_SUBPART_LETTER.match(sub_c) and sum(1 for char in body if char.isalpha()) >= 3:
                    q_num = f"Q{main_n}({sub_c})"
                    disp_num = f"{main_n}({sub_c})"
                    current_main_num = main_n

                    conf = _calculate_candidate_confidence(
                        text=body,
                        number_pattern_matched=True,
                        is_subquestion=True,
                        page_num=page_num,
                        y_ratio=0.5,
                    )

                    curr_candidate = RawCandidate(
                        q_num=q_num,
                        display_num=disp_num,
                        text=body,
                        page=page_num,
                        blocks=[b],
                        section=active_section,
                        confidence=conf,
                    )
                    candidates.append(curr_candidate)
                    continue

            # 3. Check Group Parent Header: e.g. "1. Write briefly :" or "2. Operating Systems (10 Marks)"
            if _is_group_parent_header(txt):
                m = MAIN_Q_RE.match(txt) or MAIN_Q_PREFIX_RE.match(txt)
                if m:
                    current_main_num = m.group(1)
                curr_candidate = None
                continue

            # 4. Check Main Question: e.g. "2. What is an activation function..." or "Q3. Solve equation..."
            main_m = MAIN_Q_RE.match(txt) or MAIN_Q_PREFIX_RE.match(txt)
            if main_m:
                q_num_str = main_m.group(1)
                q_rest = main_m.group(2).strip()

                # Check if q_rest contains subquestion e.g. "(a) Define..."
                sub_in = INDEPENDENT_SUB_RE.match(q_rest)
                if sub_in and VALID_SUBPART_LETTER.match(sub_in.group(1)):
                    sub_c = sub_in.group(1).lower()
                    sub_body = sub_in.group(2).strip()

                    if sum(1 for char in sub_body if char.isalpha()) >= 3:
                        full_q_num = f"Q{q_num_str}({sub_c})"
                        full_disp_num = f"{q_num_str}({sub_c})"
                        current_main_num = q_num_str

                        conf = _calculate_candidate_confidence(
                            text=sub_body,
                            number_pattern_matched=True,
                            is_subquestion=True,
                            page_num=page_num,
                            y_ratio=0.5,
                        )

                        curr_candidate = RawCandidate(
                            q_num=full_q_num,
                            display_num=full_disp_num,
                            text=sub_body,
                            page=page_num,
                            blocks=[b],
                            section=active_section,
                            confidence=conf,
                        )
                        candidates.append(curr_candidate)
                        continue

                if q_rest and sum(1 for char in q_rest if char.isalpha()) >= 3:
                    current_main_num = q_num_str
                    conf = _calculate_candidate_confidence(
                        text=q_rest,
                        number_pattern_matched=True,
                        is_subquestion=False,
                        page_num=page_num,
                        y_ratio=0.5,
                    )

                    curr_candidate = RawCandidate(
                        q_num=f"Q{q_num_str}",
                        display_num=q_num_str,
                        text=q_rest,
                        page=page_num,
                        blocks=[b],
                        section=active_section,
                        confidence=conf,
                    )
                    candidates.append(curr_candidate)
                    continue

            # 5. Check Subquestion under active main number: e.g. "a) Define deep learning." or "(b) Matrix mult..."
            sub_m = INDEPENDENT_SUB_RE.match(txt)
            if sub_m and current_main_num is not None:
                sub_c = sub_m.group(1).lower()
                sub_body = sub_m.group(2).strip()

                if VALID_SUBPART_LETTER.match(sub_c) and sum(1 for char in sub_body if char.isalpha()) >= 3:
                    full_q_num = f"Q{current_main_num}({sub_c})"
                    full_disp_num = f"{current_main_num}({sub_c})"

                    conf = _calculate_candidate_confidence(
                        text=sub_body,
                        number_pattern_matched=True,
                        is_subquestion=True,
                        page_num=page_num,
                        y_ratio=0.5,
                    )

                    curr_candidate = RawCandidate(
                        q_num=full_q_num,
                        display_num=full_disp_num,
                        text=sub_body,
                        page=page_num,
                        blocks=[b],
                        section=active_section,
                        confidence=conf,
                    )
                    candidates.append(curr_candidate)
                    continue

            # 6. Multi-line Question Continuation
            if curr_candidate is not None and curr_candidate.page == page_num:
                curr_candidate.text = f"{curr_candidate.text} {txt}"
                curr_candidate.blocks.append(b)

    # Post-process text cleanup on candidates
    for cand in candidates:
        cand.text = _clean_ocr_artifacts(cand.text)

    return candidates


async def _verify_ambiguous_candidate_with_llm(cand: RawCandidate) -> Tuple[bool, str]:
    """Targeted LLM fallback verification for ambiguous candidates only."""
    prompt = (
        "You are an AI assessment structure parser. Determine if the following text candidate is a genuine exam question requiring student response.\n"
        f"Candidate Number: {cand.display_num}\n"
        f"Candidate Text: {cand.text}\n"
        "Return strictly raw JSON format:\n"
        '{"is_question": true, "normalized_number": "' + cand.display_num + '", "text": "' + cand.text + '"}\n'
    )
    try:
        data = await asyncio.wait_for(llm_complete_json(prompt), timeout=5.0)
        if isinstance(data, dict):
            is_q = bool(data.get("is_question", True))
            norm_num = str(data.get("normalized_number", cand.display_num)).strip()
            norm_text = str(data.get("text", cand.text)).strip()
            return is_q, norm_text or cand.text
    except Exception as e:
        print(f"[QuestionExtractor] Targeted LLM check skipped/fallback ({e}), using structural decision.")

    return True, cand.text


async def extract_questions(
    blocks: List[Block],
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    doc_understanding_result: Optional[Any] = None,
    page_sizes: Optional[Dict[int, List[float]]] = None,
) -> List[Question]:
    """
    Main Entry Point: Question Paper Extraction Pipeline.
    
    Checks settings.INTELLIGENT_EXTRACTION_ENABLED (Step 11C feature flag).
    When enabled, delegates to IntelligentQuestionExtractionService.
    If an exception occurs or an empty question set is returned, gracefully falls back
    to the legacy deterministic extraction path without failing the pipeline.
    """
    if not blocks:
        return []

    # Check Step 11C Feature Flag
    if getattr(settings, "INTELLIGENT_EXTRACTION_ENABLED", True):
        try:
            from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
            service = IntelligentQuestionExtractionService()
            extraction_res = service.extract_validated_questions(
                blocks=blocks,
                document_id="qp_doc",
                doc_understanding_result=doc_understanding_result,
                page_sizes=page_sizes,
            )
            if extraction_res.questions and len(extraction_res.questions) > 0:
                return extraction_res.questions
            print("[QuestionExtractor] Step 11C returned zero questions, invoking safe legacy fallback.")
        except Exception as e:
            print(f"[QuestionExtractor] Step 11C exception ({e}), invoking safe legacy fallback.")

    return await _legacy_extract_questions(blocks, high_threshold, low_threshold)


async def _legacy_extract_questions(
    blocks: List[Block],
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
) -> List[Question]:
    """Legacy Extraction Pass."""
    high_thresh = high_threshold if high_threshold is not None else settings.QP_HIGH_CONFIDENCE_THRESHOLD
    low_thresh = low_threshold if low_threshold is not None else settings.QP_LOW_CONFIDENCE_THRESHOLD

    candidates = _extract_candidates_from_blocks(blocks)
    questions: List[Question] = []
    seen_ids: Set[str] = set()
    order_idx = 0

    parent_ids_with_subparts: Set[str] = set()
    for cand in candidates:
        if "(" in cand.q_num and cand.q_num.endswith(")"):
            main_parent_id = cand.q_num.split("(")[0]
            parent_ids_with_subparts.add(main_parent_id)

    for cand in candidates:
        if cand.q_num in seen_ids:
            continue

        if cand.q_num in parent_ids_with_subparts:
            has_q_intent = bool(SUPPORTING_INTENT_VERBS.search(cand.text)) or cand.text.endswith("?")
            if not has_q_intent:
                continue

        final_text = cand.text
        is_valid = True

        if cand.confidence >= high_thresh:
            is_valid = True
        elif cand.confidence <= low_thresh:
            is_valid = False
        else:
            is_valid, final_text = await _verify_ambiguous_candidate_with_llm(cand)

        if is_valid and final_text:
            seen_ids.add(cand.q_num)
            questions.append(
                Question(
                    id=cand.q_num,
                    number=cand.display_num,
                    text=final_text,
                    page=cand.page,
                    bbox=cand.bbox,
                    order_index=order_idx,
                    section=cand.section,
                )
            )
            order_idx += 1

    return questions
