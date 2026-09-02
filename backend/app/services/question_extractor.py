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

# Combined main + subquestion: e.g. 1(a), (1)(a), 1. (a), Q1(a), 11(a), 11-a, Q.1(a)
COMBINED_Q_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?[\(\[]?\s*(\d{1,3})\s*[\)\]]?\s*[\.\:\-\s]*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)$",
    re.IGNORECASE
)

# Main question marker: e.g. 1., 1), (1), [1], Q1, Q1., Q.1, Q 1, Question 1:, Q 1.
MAIN_Q_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?[\(\[]?\s*(\d{1,3})\s*[\)\]\.\:\-]\s*(.*)$",
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


def _has_semantic_question_structure(doc_understanding_result: Optional[Any]) -> bool:
    """True only when the VLM / semantic graph already produced document structure."""
    if doc_understanding_result is None:
        return False

    graph = getattr(doc_understanding_result, "structure_graph", None)
    if graph and getattr(graph, "nodes", None):
        return any(getattr(node, "role", None) == "QUESTION" for node in graph.nodes.values())

    vlm_understandings = getattr(doc_understanding_result, "vlm_page_understandings", []) or []
    if any(getattr(u, "structures", None) for u in vlm_understandings):
        return True

    if getattr(doc_understanding_result, "regions", None):
        return any(getattr(r, "region_type", None) == "QUESTION" for r in doc_understanding_result.regions)

    return False


def pil_image_to_b64(img: Any) -> str:
    import io
    import base64
    buf = io.BytesIO()
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_UUID_RE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.IGNORECASE)
_VALID_QNUM_RE = re.compile(r"^\d{1,3}(?:[\(\[]?[a-z]{1,2}[\)\]]?|[\-\.][ivxlIVXL]{1,4})?$", re.IGNORECASE)


def _is_valid_question_number(s: str) -> bool:
    """
    Returns True if s looks like a real question number from a question paper.
    Accepts: "1", "12", "3(a)", "5(i)", "10(ii)", "2b", "Q3"
    Rejects: UUID hex strings, empty strings, purely alphabetic strings.
    """
    if not s or not s.strip():
        return False
    t = s.strip()
    # Reject UUID-like strings the VLM sometimes hallucinates
    if _UUID_RE.match(t):
        return False
    # Strip leading Q/q prefix before checking
    if t.lower().startswith("q"):
        t = t[1:].strip().lstrip(".")
    return bool(_VALID_QNUM_RE.match(t))


async def extract_questions_vlm(qp_images_dict: Dict[int, Any]) -> List[Question]:
    """
    Direct 100% VLM Question Paper Extractor.
    Inspects Question Paper page images directly using Gemini VLM, extracting structured questions.

    Key fixes vs old implementation:
    - DOES NOT strip subquestion letters ("3(a)" preserved as "3(a)", not collapsed to "3")
    - Captures bbox coordinates for each question
    - Captures parent_question_number to build hierarchy
    - Captures marks where visible
    - Strengthened prompt: explicitly excludes metadata, titles, instructions, section headers
    """
    from app.services.llm_provider import llm_complete_multimodal_with_metadata
    import json
    import re

    all_questions: List[Question] = []
    order_idx = 0

    prompt = """You are an expert exam paper analyzer. You are looking at a QUESTION PAPER image.
Your task: extract every ASSESSABLE QUESTION that a student must answer.

Return ONLY valid JSON in this exact structure (no markdown, no explanation):
{
  "questions": [
    {
      "number": "1",
      "text": "What is the SI unit of force?",
      "max_marks": 2.0,
      "question_type": "SHORT_ANSWER",
      "options": [],
      "bbox": {"x": 120, "y": 430, "width": 860, "height": 45},
      "parent_question_number": null
    },
    {
      "number": "5(a)",
      "text": "Define photosynthesis.",
      "max_marks": 2.0,
      "question_type": "SHORT_ANSWER",
      "options": [],
      "bbox": {"x": 120, "y": 650, "width": 860, "height": 45},
      "parent_question_number": "5"
    },
    {
      "number": "8",
      "text": "Which planet is known as the Red Planet?",
      "max_marks": 1.0,
      "question_type": "MCQ",
      "options": ["A. Earth", "B. Mars", "C. Jupiter", "D. Venus"],
      "bbox": {"x": 120, "y": 900, "width": 860, "height": 90},
      "parent_question_number": null
    }
  ]
}

FIELD RULES:
- "number": Copy the EXACT question number as printed. PRESERVE subpart letters.
  CORRECT: "1", "2", "3(a)", "3(b)", "5(i)", "5(ii)", "Q4", "10"
  WRONG: Do NOT simplify "3(a)" → "3". Do NOT use UUIDs or hex IDs.
- "text": The full question text that the student reads and responds to.
- "max_marks": Numeric marks for this question/subpart. Look for (2), [5 Marks], etc. Use 0 if not visible.
- "question_type": "MCQ" | "SHORT_ANSWER" | "LONG_ANSWER" | "NUMERICAL"
- "options": For MCQ, list each option string including its label ("A. Earth", "B. Mars", ...). Empty [] for non-MCQ.
- "bbox": Pixel bounding box of the question text in this image.
  {"x": pixels_from_left, "y": pixels_from_top, "width": pixel_width, "height": pixel_height}
  Estimate based on where you visually see the text. Top-left origin.
- "parent_question_number": For subquestions, write the parent number (e.g. "5" for "5(a)"). null for top-level.

WHAT TO INCLUDE:
- Numbered questions the student must answer
- MCQ questions with their options
- Subquestions and subparts — each as a SEPARATE entry, with parent_question_number set
- Multi-line questions (combine all lines into one "text" field)

WHAT TO EXCLUDE (these are NOT questions — do NOT include them):
- Document title: "SCIENCE QUESTION PAPER", "FINAL EXAMINATION"
- Metadata: class, school, date, time, maximum marks, subject code
- Student fields: "Name: _____", "Roll No: _____"
- Instructions: "Attempt any four", "All questions are compulsory", "Read carefully"
- Section headers: "SECTION A", "PART B", "GROUP I"
- Page numbers, watermarks, footers
- Parent-header-only items like "5. Answer the following:" that have NO answerable content themselves
  (their subquestions like 5(a), 5(b) etc. should be included with parent_question_number="5")
"""

    for page_num in sorted(qp_images_dict.keys()):
        img = qp_images_dict[page_num]
        b64_img = pil_image_to_b64(img)

        # Track actual image dimensions for bbox validation
        img_w = getattr(img, "width", 1240) if hasattr(img, "width") else 1240
        img_h = getattr(img, "height", 1754) if hasattr(img, "height") else 1754

        try:
            raw_res, meta = await llm_complete_multimodal_with_metadata(
                prompt=prompt,
                image_b64=b64_img,
                mime_type="image/jpeg",
                purpose=f"vlm_question_extraction_p{page_num}",
            )

            cleaned_json = raw_res.strip()
            if "```json" in cleaned_json:
                cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_json:
                cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

            m_json = re.search(r"\{.*\}", cleaned_json, re.DOTALL)
            if m_json:
                cleaned_json = m_json.group(0)

            data = json.loads(cleaned_json)
            raw_qs = data.get("questions", []) if isinstance(data, dict) else []

            for q_dict in raw_qs:
                # ----------------------------------------------------------------
                # CRITICAL FIX: DO NOT strip subquestion letters.
                # Old code: m_digit = re.search(r"(\d{1,3})", raw_num)
                #           clean_num = m_digit.group(1)  ← DESTROYED "3(a)" → "3"
                # New code: Preserve the full number the VLM returns.
                # ----------------------------------------------------------------
                raw_num = str(q_dict.get("number", "")).strip()
                if not _is_valid_question_number(raw_num):
                    # Fallback: try to extract number+subpart together
                    m_fallback = re.search(
                        r"(\d{1,3}(?:[\(\[][a-z]{1,3}[\)\]])?)",
                        raw_num, re.IGNORECASE
                    )
                    raw_num = m_fallback.group(1) if m_fallback else str(order_idx + 1)

                # Strip leading Q/q prefix if present ("Q3(a)" → "3(a)")
                clean_num = raw_num.strip()
                if clean_num.lower().startswith("q"):
                    clean_num = clean_num[1:].strip().lstrip(".")

                q_text = str(q_dict.get("text", "")).strip()
                try:
                    m_marks = float(q_dict.get("max_marks", 0.0) or 0.0)
                except (TypeError, ValueError):
                    m_marks = 0.0

                q_type_raw = str(q_dict.get("question_type", "SHORT_ANSWER")).upper().strip()
                # Normalize to known types
                q_type = q_type_raw if q_type_raw in (
                    "MCQ", "SHORT_ANSWER", "LONG_ANSWER", "NUMERICAL", "SUBQUESTION", "UNKNOWN"
                ) else "SHORT_ANSWER"

                opts = q_dict.get("options", []) or []
                if not isinstance(opts, list):
                    opts = [str(opts)]
                opts = [str(o).strip() for o in opts if str(o).strip()]

                # Extract bbox from VLM response (preserve coordinates for highlighting)
                bbox_raw = q_dict.get("bbox") or {}
                q_bbox: Optional[BBox] = None
                if isinstance(bbox_raw, dict):
                    try:
                        bx = float(bbox_raw.get("x", 0) or 0)
                        by = float(bbox_raw.get("y", 0) or 0)
                        bw = float(bbox_raw.get("width", 0) or 0)
                        bh = float(bbox_raw.get("height", 0) or 0)
                        if bw > 0 and bh > 0:
                            # Clamp to image bounds
                            bx = max(0.0, min(bx, img_w - 1))
                            by = max(0.0, min(by, img_h - 1))
                            bw = max(1.0, min(bw, img_w - bx))
                            bh = max(1.0, min(bh, img_h - by))
                            q_bbox = BBox(x=bx, y=by, width=bw, height=bh)
                    except (TypeError, ValueError):
                        q_bbox = None

                # Capture parent question number for subquestion hierarchy
                parent_q_num_raw = q_dict.get("parent_question_number") or None
                parent_q_id: Optional[str] = None
                if parent_q_num_raw and str(parent_q_num_raw).strip():
                    pn = str(parent_q_num_raw).strip()
                    if pn.lower().startswith("q"):
                        pn = pn[1:].strip().lstrip(".")
                    if pn and _is_valid_question_number(pn):
                        parent_q_id = f"Q{pn}"

                if q_text:
                    q_id = f"Q{clean_num}"
                    q_obj = Question(
                        id=q_id,
                        number=clean_num,
                        text=q_text,
                        page=page_num,
                        bbox=q_bbox,
                        order_index=order_idx,
                        question_type=q_type,  # type: ignore
                        options=opts,
                        parent_question_id=parent_q_id,
                        extraction_confidence=1.0 if meta.get("vlm_result") == "SUCCESS" else 0.5,
                    )
                    all_questions.append(q_obj)
                    order_idx += 1
                    print(
                        f"[VLMQuestionExtractor] Q{clean_num}"
                        f" (parent={parent_q_id}, type={q_type}, marks={m_marks})"
                        f" page={page_num}: '{q_text[:55]}'"
                    )

        except Exception as e:
            print(f"[VLMQuestionExtractor] Error parsing VLM response on page {page_num}: {e}")

    # Natural sort: page → leading integer → subpart letter → original order
    def _sort_key(q: Question):
        m = re.match(r"^\s*(\d{1,3})", q.number)
        val = int(m.group(1)) if m else 999
        sub_m = re.search(r"[\(\[]([a-z]{1,2}|i{1,3}v?|vi{0,3})[\)\]]", q.number, re.IGNORECASE)
        sub_val = ord(sub_m.group(1).lower()[0]) if sub_m else -1
        return (q.page, val, sub_val, q.order_index)

    all_questions.sort(key=_sort_key)
    for idx, q in enumerate(all_questions):
        q.order_index = idx

    print(f"[VLMQuestionExtractor] Total extracted: {len(all_questions)} questions across {len(qp_images_dict)} page(s)")
    return all_questions


async def extract_questions(
    blocks: List[Block],
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    doc_understanding_result: Optional[Any] = None,
    page_sizes: Optional[Dict[int, List[float]]] = None,
) -> List[Question]:
    """
    Strict VLM-first extraction order:
    1. require semantic graph / VLM structure
    2. never invoke OCR/regex fallback in production mode
    3. fail explicitly if VLM does not provide the structure
    """
    if not blocks:
        return []

    semantic_present = _has_semantic_question_structure(doc_understanding_result)
    if getattr(settings, "STRICT_VLM_ONLY_MODE", True) and not semantic_present:
        raise RuntimeError("OCR/regex fallback is disabled. VLM/LLM semantic structure is required.")

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
        except Exception as e:
            print(f"[QuestionExtractor] Semantic extraction failed: {e}")
            raise RuntimeError("VLM/LLM semantic extraction failed; OCR/regex fallback is disabled.") from e

    if not semantic_present:
        raise RuntimeError("OCR/regex fallback is disabled. VLM/LLM semantic structure is required.")

    return []


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
