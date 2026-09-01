"""
Answer Sheet Intelligence & Structured Region Extraction Engine.

Generalized Architecture:
- Strict separation of Modality (handwritten / printed / mixed / unknown) from Role (question_reference / student_question_anchor / student_answer / metadata / etc.).
- Distinguishes Printed Question References from Student Question Anchors (solving the Q1(j) false-answer bug).
- Dynamic Anchor Detection & False-Positive Suppression (handles arbitrary subjects, layouts, fonts, page sizes, and numbering styles).
- Spatial Column-Aware Matching for multi-column printed references & student answers.
- Dynamic Multi-Page Continuation Linking across page boundaries.
- Preserves Visual Geometry & Unanchored Answers without premature "unanswered" declarations.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from typing import List, Optional, Dict, Set, Tuple, Any
from app.models.schemas import (
    Block, BlockModality, BlockRole, Region, BBox, QuestionAnchor, AnswerRegion,
    PageAnalysis, StructuredAnswerSheet, AnswerCandidate,
)
from app.services.page_intelligence import analyze_pages
from app.services.llm_provider import llm_complete_json

# --- Anchor Regex Patterns ---

# Combined anchor with parenthesis: e.g. "1(a).", "1(a)", "Q1(a)", "Ans 1(a).", "1(j)."
DIRECT_PAREN_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\s\-_]*\(([a-zA-Z0-9]{1,2})\)\s*[\.\):]?\s*(.*)$",
    re.IGNORECASE,
)

# Combined anchor with dot/dash: e.g. "1.a.", "1-a", "Q1.a", "1a."
DIRECT_DOT_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\-_]\s*([a-zA-Z0-9]{1,2})\s*[\.\):]\s*(.*)$",
    re.IGNORECASE,
)

# Main anchor: e.g. "Q1.", "Q1", "Q.1", "Q 1", "1.", "1)", "01", "Q7.", "Q7"
MAIN_ANCHOR_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})\s*[\.\):]?\s*(.*)$",
    re.IGNORECASE,
)

# Subpart anchor under active group parent: e.g. "a) Operating System...", "(b) Bias..."
SUBPART_ANCHOR_RE = re.compile(
    r"^\s*(?:\(([a-zA-Z0-9]{1,2})\)|([a-zA-Z0-9]{1,2})[\.\):])\s+(.*)$",
    re.IGNORECASE,
)

# Continuation Anchor Header e.g. "Q7. Back propagation algorithm (continued)" or "Q8. (continued)"
CONTINUATION_HEADER_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*(?:\(([a-zA-Z0-9]{1,2})\))?\.?\s*.*?\b(?:continued|\(continued\))\b",
    re.IGNORECASE,
)

# Explicit continuation notice line e.g. "Q7 continues on the next page"
CONTINUATION_FOOTER_RE = re.compile(
    r"^\s*Q?\d{1,3}.*?\bcontinues\b.*?\bpage\b",
    re.IGNORECASE,
)

# Explicit unanswered notice line e.g. "[intentionallyunanswered ]" or "left blank"
UNANSWERED_NOTICE_RE = re.compile(
    r"^\s*(?:\[?\s*intentionally\s*unanswered\s*\]?|\[?\s*unanswered\s*\]?|not\s*attempted|left\s*blank|.*intentionally\s*left\s*unanswered.*)\s*$",
    re.IGNORECASE,
)

# Supporting Intent Verbs (Used to evaluate if a printed line is a question reference prompt)
QUESTION_PROMPT_VERBS = re.compile(
    r"\b(?:what|why|how|explain|discuss|calculate|derive|prove|compare|define|state|find|show|list|describe|illustrate|evaluate|analyze|differentiate|determine|solve|convert|identify|construct|write|compute|trace|sketch|distinguish|design)\b",
    re.IGNORECASE,
)

# Noise and Header/Footer Patterns
HEADER_FOOTER_RE = re.compile(
    r"^\s*(?:SECTION\s*-\s*[A-Z0-9]+\s*\(continued\)|ANSWERSHEET|Page\s*\d+\s*(?:of|/)\s*\d+|End\s+of\s+the\s+Question\s+Paper|\d{1,2}\s*[\/\\]\s*\d{1,2}\s*$)",
    re.IGNORECASE,
)

WATERMARK_NOISE_RE = re.compile(
    r"brpaper\.com|aglasem\.com|^\d{1,2}\s*\|\s*(?:m\-)?",
    re.IGNORECASE,
)


def _clean_mark_annotations(text: str) -> str:
    """Strips trailing mark annotations like '[2 x 5 = 10 Marks]' or '(5 Marks)'."""
    return re.sub(r"\[\s*.*?\s*Marks?\s*\]|\(\s*\d+\s*Marks?\s*\)", "", text, flags=re.IGNORECASE).strip()


def _is_false_positive_number(text: str, main_num: str) -> bool:
    """
    Suppresses false-positive numbers such as dates (2026), registration numbers (123456),
    marks annotations (2 marks), page pagination counters (1/8, 2/8), or pure math values (0.40).
    """
    t = text.strip().lower()

    if re.match(r"^\d{1,2}\s*[\/\\]\s*\d{1,2}\s*$", t):
        return True

    if CONTINUATION_FOOTER_RE.match(t):
        return True

    if main_num == "0" or t.startswith("0."):
        return True

    if len(main_num) >= 4 and not (t.startswith("q") or t.startswith("ans")):
        return True

    if ("marks" in t or "page " in t or "total" in t) and not (t.startswith("q") or t.startswith("ans")):
        return True

    if re.match(r"^\d{4}[\-\/]\d{1,2}[\-\/]\d{1,2}$", t):
        return True

    return False


def _is_question_reference_prompt(text: str, rest_text: str) -> bool:
    """
    Structural check: Determines if a line is a printed question paper prompt
    reproduced on the answer sheet (e.g. '1(g). Recursive neural networks apply...' or
    '1(j). Discuss in brief about Sigmoid belief networks.') vs a student question heading
    (e.g. '1(a).' or 'Q7.') or a typed answer block.
    """
    t = text.strip()
    r = rest_text.strip()

    # Integrated typed question + answer (e.g. "Q1. What is X? X is a neural network...")
    if "?" in r:
        _, ans_part = r.split("?", 1)
        if len(ans_part.strip()) > 15:
            return False

    # Typed student answer after anchor (e.g. "Q1. Backpropagation computes the gradient of...")
    if len(r) > 40 and not QUESTION_PROMPT_VERBS.search(r) and not t.endswith("?"):
        return False

    # Long text (>25 chars) after an anchor containing prompt verbs or ending in ?
    if len(r) > 25 and (QUESTION_PROMPT_VERBS.search(r) or t.endswith("?")):
        return True

    # Printed cover table entries where full question text is reproduced
    if len(t) > 40 and QUESTION_PROMPT_VERBS.search(t):
        return True

    return False


def _detect_anchors_and_references_in_blocks(
    blocks: List[Block],
) -> Tuple[Dict[int, List[QuestionAnchor]], Dict[int, List[QuestionAnchor]]]:
    """
    Detects and normalizes student question anchors and printed question references across blocks.
    Returns:
      anchors_by_page: Dict[page_num, List[QuestionAnchor(role="student_question_anchor")]]
      references_by_page: Dict[page_num, List[QuestionAnchor(role="question_reference")]]
    """
    anchors_by_page: Dict[int, List[QuestionAnchor]] = {}
    references_by_page: Dict[int, List[QuestionAnchor]] = {}
    current_main_num: Optional[str] = None
    reading_order_idx = 0

    blocks_by_page: Dict[int, List[Block]] = {}
    for b in blocks:
        blocks_by_page.setdefault(b.page, []).append(b)

    for page_num in sorted(blocks_by_page.keys()):
        p_blocks = sorted(blocks_by_page[page_num], key=lambda b: (b.bbox.y, b.bbox.x))

        for b in p_blocks:
            txt = b.text.strip()
            if not txt or HEADER_FOOTER_RE.match(txt) or WATERMARK_NOISE_RE.search(txt) or UNANSWERED_NOTICE_RE.match(txt):
                if UNANSWERED_NOTICE_RE.match(txt):
                    b.role = "noise"
                continue

            if CONTINUATION_FOOTER_RE.match(txt):
                b.role = "header_footer"
                continue

            cleaned_txt = _clean_mark_annotations(txt)

            # 1. Combined Paren Anchor: e.g. "1(a)." or "1(j). Discuss in brief..."
            #    BUT NOT MCQ-answer lines like "Q1. (D) Combustion..." where the option
            #    letter follows a period+space and represents the chosen option, not a
            #    subquestion label.  We detect this by checking whether the text between
            #    the question number and the opening parenthesis contains a period followed
            #    by at least one whitespace character (structurally: ". (").
            m_paren = DIRECT_PAREN_RE.match(cleaned_txt)
            if m_paren:
                main_n = m_paren.group(1)
                sub_c = m_paren.group(2).lower()
                rest = m_paren.group(3).strip()

                # Determine whether the separator is ". (" (MCQ-answer style) or
                # directly adjacent "(" (true subquestion style).  We find the
                # position of the matched number in the cleaned string and inspect
                # what comes immediately after it.
                num_end_pos = cleaned_txt.index(main_n) + len(main_n)
                after_num = cleaned_txt[num_end_pos:]
                # If the text after the number begins with optional whitespace then
                # a period then whitespace before the opening paren, this is an
                # MCQ-answer selection, not a subquestion anchor.
                _is_mcq_option_separator = bool(
                    re.match(r"^\s*\.\s+\(", after_num)
                )

                if _is_mcq_option_separator:
                    # Fall through — let the MAIN_ANCHOR_RE branch below handle
                    # this block as a plain main-question anchor (Q{n}), which
                    # keeps the full "(D) Combustion..." text as the answer body.
                    pass
                elif not _is_false_positive_number(txt, main_n):
                    norm_anchor = f"Q{main_n}({sub_c})"
                    current_main_num = main_n

                    is_q_ref = _is_question_reference_prompt(txt, rest)
                    role_type = "question_reference" if is_q_ref else "student_question_anchor"

                    anchor_obj = QuestionAnchor(
                        anchor=norm_anchor,
                        original_text=txt,
                        role=role_type,
                        page=page_num,
                        bbox=b.bbox,
                        confidence=0.98,
                        reading_order=reading_order_idx,
                    )
                    reading_order_idx += 1

                    if is_q_ref:
                        references_by_page.setdefault(page_num, []).append(anchor_obj)
                        b.role = "question_reference"
                    else:
                        anchors_by_page.setdefault(page_num, []).append(anchor_obj)
                        b.role = "student_question_anchor"
                    continue

            # 2. Combined Dot Anchor: e.g. "1.a." or "1-a."
            m_dot = DIRECT_DOT_RE.match(cleaned_txt)
            if m_dot:
                main_n = m_dot.group(1)
                sub_c = m_dot.group(2).lower()
                rest = m_dot.group(3).strip()

                if not _is_false_positive_number(txt, main_n):
                    norm_anchor = f"Q{main_n}({sub_c})"
                    current_main_num = main_n

                    is_q_ref = _is_question_reference_prompt(txt, rest)
                    role_type = "question_reference" if is_q_ref else "student_question_anchor"

                    anchor_obj = QuestionAnchor(
                        anchor=norm_anchor,
                        original_text=txt,
                        role=role_type,
                        page=page_num,
                        bbox=b.bbox,
                        confidence=0.98,
                        reading_order=reading_order_idx,
                    )
                    reading_order_idx += 1

                    if is_q_ref:
                        references_by_page.setdefault(page_num, []).append(anchor_obj)
                        b.role = "question_reference"
                    else:
                        anchors_by_page.setdefault(page_num, []).append(anchor_obj)
                        b.role = "student_question_anchor"
                    continue

            # 3. Continuation Anchor Header: e.g. "Q7. Back propagation algorithm (continued)"
            m_cont = CONTINUATION_HEADER_RE.match(cleaned_txt)
            if m_cont:
                main_n = m_cont.group(1)
                sub_c = m_cont.group(2).lower() if m_cont.group(2) else None
                norm_anchor = f"Q{main_n}({sub_c})" if sub_c else f"Q{main_n}"

                anchor_obj = QuestionAnchor(
                    anchor=norm_anchor,
                    original_text=txt,
                    role="student_question_anchor",
                    page=page_num,
                    bbox=b.bbox,
                    confidence=0.95,
                    reading_order=reading_order_idx,
                )
                reading_order_idx += 1
                anchors_by_page.setdefault(page_num, []).append(anchor_obj)
                b.role = "student_question_anchor"
                continue

            # 4. Main Question Anchor: e.g. "Q7.", "Q7", "7."
            m_main = MAIN_ANCHOR_RE.match(cleaned_txt)
            if m_main:
                main_num = m_main.group(1)
                rest = m_main.group(2).strip()

                is_explicit_q = txt.lower().startswith("q") or txt.lower().startswith("ans")
                if is_explicit_q or (len(rest) < 60 and not _is_false_positive_number(txt, main_num)):
                    norm_anchor = f"Q{main_num}"
                    current_main_num = main_num

                    is_q_ref = _is_question_reference_prompt(txt, rest)
                    role_type = "question_reference" if is_q_ref else "student_question_anchor"

                    anchor_obj = QuestionAnchor(
                        anchor=norm_anchor,
                        original_text=txt,
                        role=role_type,
                        page=page_num,
                        bbox=b.bbox,
                        confidence=0.95 if is_explicit_q else 0.85,
                        reading_order=reading_order_idx,
                    )
                    reading_order_idx += 1

                    if is_q_ref:
                        references_by_page.setdefault(page_num, []).append(anchor_obj)
                        b.role = "question_reference"
                    else:
                        anchors_by_page.setdefault(page_num, []).append(anchor_obj)
                        b.role = "student_question_anchor"
                    continue

            # 5. Subpart Anchor under active main header: e.g. "a) Operating System..."
            m_sub = SUBPART_ANCHOR_RE.match(cleaned_txt)
            if m_sub and current_main_num is not None:
                sub_char = (m_sub.group(1) or m_sub.group(2)).lower()
                rest = m_sub.group(3).strip()
                norm_anchor = f"Q{current_main_num}({sub_char})"

                is_q_ref = _is_question_reference_prompt(txt, rest)
                role_type = "question_reference" if is_q_ref else "student_question_anchor"

                anchor_obj = QuestionAnchor(
                    anchor=norm_anchor,
                    original_text=txt,
                    role=role_type,
                    page=page_num,
                    bbox=b.bbox,
                    confidence=0.92,
                    reading_order=reading_order_idx,
                )
                reading_order_idx += 1

                if is_q_ref:
                    references_by_page.setdefault(page_num, []).append(anchor_obj)
                    b.role = "question_reference"
                else:
                    anchors_by_page.setdefault(page_num, []).append(anchor_obj)
                    b.role = "student_question_anchor"
                continue

    return anchors_by_page, references_by_page


def _classify_block_roles_and_modalities(blocks: List[Block], metadata_pages: Set[int]) -> None:
    """
    Categorizes every OCR block into a role and modality:
    - BlockRoles: student_question_anchor, question_reference, student_answer, page_metadata, header_footer, noise, visual_element.
    - BlockModalities: printed, handwritten, mixed, unknown.
    """
    for b in blocks:
        if b.role in ("student_question_anchor", "question_reference", "noise"):
            continue

        txt = b.text.strip()
        if not txt:
            b.role = "noise"
            continue

        if b.page in metadata_pages:
            b.role = "page_metadata"
            continue

        if WATERMARK_NOISE_RE.search(txt) or UNANSWERED_NOTICE_RE.match(txt):
            b.role = "noise"
            continue

        if HEADER_FOOTER_RE.match(txt) or CONTINUATION_FOOTER_RE.match(txt):
            b.role = "header_footer"
            continue

        if b.confidence < 0.30:
            b.role = "visual_element"
            continue

        b.role = "student_answer"


def _detect_page_column_clusters(blocks: List[Block], page_w: float) -> List[float]:
    """
    Dynamically detects horizontal column centroids on a page using normalized X-center positions.
    Supports 1-column, 2-column, multi-column, and centered layouts without hardcoded column counts.
    """
    if not blocks or page_w <= 0:
        return [0.5]

    norm_centers = sorted(((b.bbox.x + b.bbox.width / 2.0) / page_w) for b in blocks)

    clusters: List[List[float]] = []
    for x in norm_centers:
        if not clusters:
            clusters.append([x])
        else:
            if x - clusters[-1][-1] < 0.22:  # 22% of page width clustering threshold
                clusters[-1].append(x)
            else:
                clusters.append([x])

    column_centroids = [sum(c) / len(c) for c in clusters]
    return column_centroids


def _evaluate_spatial_reference_match(
    ref: QuestionAnchor,
    block: Block,
    b_order_idx: int,
    page_w: float,
    page_h: float,
    column_centroids: List[float],
) -> float:
    """
    Multi-Signal Spatial Alignment Evaluator (Resolution & Layout Independent).
    
    Combines:
    1. Relative Horizontal Alignment (Normalized X & Column Cluster Matching)
    2. Normalized Vertical Relationship (Block must be below or level with reference)
    3. Reading Order & Spatial Proximity
    4. Layout Containment Score
    
    Returns a normalized confidence score S in [0.0, 1.0].
    """
    if page_w <= 0 or page_h <= 0:
        return 0.0

    ref_nx = (ref.bbox.x + ref.bbox.width / 2.0) / page_w
    ref_ny = ref.bbox.y / page_h

    b_nx = (block.bbox.x + block.bbox.width / 2.0) / page_w
    b_ny = block.bbox.y / page_h

    # 1. Vertical relationship signal: block must be below or level with reference anchor
    dy = b_ny - ref_ny
    if dy < -0.015:  # block is significantly above the anchor -> invalid match
        return 0.0

    s_vert = max(0.0, 1.0 - 2.5 * dy)

    # 2. Horizontal Column Alignment signal
    if len(column_centroids) > 1:
        ref_col = min(range(len(column_centroids)), key=lambda i: abs(column_centroids[i] - ref_nx))
        b_col = min(range(len(column_centroids)), key=lambda i: abs(column_centroids[i] - b_nx))

        if ref_col != b_col:
            # Belongs to a different column cluster -> zero score across columns!
            return 0.0
        s_horiz = 1.0
    else:
        # Single column / centered layout: normalized X center difference
        dx = abs(ref_nx - b_nx)
        s_horiz = max(0.0, 1.0 - 3.0 * dx)

    # 3. Reading order proximity signal
    d_order = max(0, b_order_idx - ref.reading_order)
    s_order = max(0.0, 1.0 - 0.05 * d_order)

    # Multi-signal weighted score calculation
    score = 0.45 * s_horiz + 0.40 * s_vert + 0.15 * s_order
    return score


def process_answer_sheet(
    blocks: List[Block],
    num_pages: int,
    page_sizes: Optional[List[Tuple[int, int]]] = None,
) -> StructuredAnswerSheet:
    """
    Main Entry Point: Generalized Answer Sheet Intelligence Pipeline.
    
    Produces StructuredAnswerSheet containing:
    - page_analyses (Dict[page_num, PageAnalysis])
    - question_anchors (List[QuestionAnchor(role="student_question_anchor")])
    - question_references (List[QuestionAnchor(role="question_reference")])
    - answer_regions (List[AnswerRegion] - student response regions only)
    - unanchored_regions (List[AnswerRegion] - unanchored student work)
    - metadata_blocks & noise_blocks (preserved visual geometry)
    """
    if page_sizes is None:
        page_sizes = [[1000, 1400] for _ in range(num_pages)]

    sizes_list = [[int(w), int(h)] for (w, h) in page_sizes]

    # 1. Question Anchor & Question Reference Detection
    anchors_by_page, references_by_page = _detect_anchors_and_references_in_blocks(blocks)

    all_anchors: List[QuestionAnchor] = []
    for p in sorted(anchors_by_page.keys()):
        all_anchors.extend(anchors_by_page[p])

    all_references: List[QuestionAnchor] = []
    for p in sorted(references_by_page.keys()):
        all_references.extend(references_by_page[p])

    # Combine all anchors for page intelligence analysis
    all_page_anchors: Dict[int, List[QuestionAnchor]] = {}
    for p in range(1, num_pages + 1):
        all_page_anchors[p] = anchors_by_page.get(p, []) + references_by_page.get(p, [])

    # 2. Page-level Intelligence Analysis
    page_analyses, metadata_pages = analyze_pages(blocks, num_pages, all_page_anchors)

    # 3. Block Role and Modality Classification
    _classify_block_roles_and_modalities(blocks, metadata_pages)

    # Separate preserved non-answer blocks
    metadata_blocks = [b for b in blocks if b.role in ("page_metadata", "header_footer")]
    printed_question_blocks = [b for b in blocks if b.role == "question_reference"]
    noise_blocks = [b for b in blocks if b.role == "noise"]

    # 4. Logical Answer Region Segmentation & Dynamic Multi-Page Continuation
    answer_regions: List[AnswerRegion] = []
    unanchored_regions: List[AnswerRegion] = []
    active_anchor_regions: Dict[str, AnswerRegion] = {}

    blocks_by_page: Dict[int, List[Block]] = {}
    for b in blocks:
        if b.role in ("student_answer", "student_question_anchor", "question_reference", "visual_element"):
            blocks_by_page.setdefault(b.page, []).append(b)

    region_idx = 0

    for page_num in range(1, num_pages + 1):
        if page_num in metadata_pages:
            continue

        p_blocks = sorted(blocks_by_page.get(page_num, []), key=lambda b: (b.bbox.y, b.bbox.x))
        p_anchors = anchors_by_page.get(page_num, [])
        p_anchors_by_y = {round(a.bbox.y, 1): a for a in p_anchors}

        p_refs = references_by_page.get(page_num, [])
        p_refs_by_y = {round(r.bbox.y, 1): r for r in p_refs}

        # Determine resolution-independent normalized page dimensions
        if page_sizes and page_num <= len(page_sizes) and page_sizes[page_num - 1][0] > 0:
            page_w, page_h = float(page_sizes[page_num - 1][0]), float(page_sizes[page_num - 1][1])
        else:
            page_w = max((b.bbox.x + b.bbox.width for b in p_blocks), default=1000.0)
            page_h = max((b.bbox.y + b.bbox.height for b in p_blocks), default=1400.0)

        # Detect dynamic column layout clusters on page
        column_centroids = _detect_page_column_clusters(p_blocks, page_w)

        pending_ref_anchors: List[QuestionAnchor] = []
        active_region: Optional[AnswerRegion] = None

        for b_idx, b in enumerate(p_blocks):
            b_y = round(b.bbox.y, 1)

            # 4A. Check if block is a Printed Question Reference (e.g. 1(a). Define... vs 1(j). Discuss...)
            if b_y in p_refs_by_y or b.role == "question_reference":
                matched_ref = p_refs_by_y.get(b_y)
                if matched_ref:
                    pending_ref_anchors.append(matched_ref)
                    active_region = None
                    continue

            # 4B. Check if block is a Student Question Anchor Heading (e.g. Q7. or 1(a).)
            if b_y in p_anchors_by_y or b.role == "student_question_anchor":
                matched_anchor = p_anchors_by_y.get(b_y)
                anchor_name = matched_anchor.anchor if matched_anchor else None
                pending_ref_anchors.clear()

                # Multi-page Continuation Check
                if anchor_name and anchor_name in active_anchor_regions:
                    active_region = active_anchor_regions[anchor_name]
                    if page_num not in active_region.pages:
                        active_region.pages.append(page_num)
                        active_region.is_continuation = True
                    active_region.blocks.append(b)
                    active_region.regions.append(Region(page=page_num, bbox=b.bbox))
                    continue

                # Start new AnswerRegion for this student question anchor
                active_region = AnswerRegion(
                    answer_id=f"region_{uuid.uuid4().hex[:8]}",
                    question_anchor=anchor_name,
                    pages=[page_num],
                    regions=[Region(page=page_num, bbox=b.bbox)],
                    text=b.text.strip(),
                    blocks=[b],
                    reading_order=region_idx,
                    confidence=matched_anchor.confidence if matched_anchor else 0.9,
                )
                region_idx += 1
                answer_regions.append(active_region)
                if anchor_name:
                    active_anchor_regions[anchor_name] = active_region
                continue

            # 4C. Student Answer Text or Visual Block
            if b.role in ("student_answer", "visual_element"):
                # Evaluate multi-signal normalized spatial alignment score across pending reference anchors
                if pending_ref_anchors:
                    scored_candidates = [
                        (ref, _evaluate_spatial_reference_match(ref, b, b_idx, page_w, page_h, column_centroids))
                        for ref in pending_ref_anchors
                    ]
                    valid_candidates = [(ref, score) for ref, score in scored_candidates if score >= 0.35]

                    if valid_candidates:
                        best_ref, best_score = max(valid_candidates, key=lambda pair: pair[1])
                        ref_name = best_ref.anchor
                        pending_ref_anchors.remove(best_ref)

                        if ref_name in active_anchor_regions:
                            active_region = active_anchor_regions[ref_name]
                            if page_num not in active_region.pages:
                                active_region.pages.append(page_num)
                                active_region.is_continuation = True
                            active_region.text = f"{active_region.text} {b.text.strip()}".strip()
                            active_region.blocks.append(b)
                            active_region.regions.append(Region(page=page_num, bbox=b.bbox))
                        else:
                            active_region = AnswerRegion(
                                answer_id=f"region_{uuid.uuid4().hex[:8]}",
                                question_anchor=ref_name,
                                pages=[page_num],
                                regions=[Region(page=page_num, bbox=b.bbox)],
                                text=b.text.strip(),
                                blocks=[b],
                                reading_order=region_idx,
                                confidence=0.92,
                            )
                            region_idx += 1
                            answer_regions.append(active_region)
                            active_anchor_regions[ref_name] = active_region
                        continue

                if active_region is not None:
                    if page_num not in active_region.pages:
                        active_region.pages.append(page_num)
                        active_region.is_continuation = True

                    active_region.text = f"{active_region.text} {b.text.strip()}".strip()
                    active_region.blocks.append(b)
                    active_region.regions.append(Region(page=page_num, bbox=b.bbox))
                else:
                    # Unanchored student work (written without explicit Q-number)
                    active_region = AnswerRegion(
                        answer_id=f"region_{uuid.uuid4().hex[:8]}",
                        question_anchor=None,
                        pages=[page_num],
                        regions=[Region(page=page_num, bbox=b.bbox)],
                        text=b.text.strip(),
                        blocks=[b],
                        reading_order=region_idx,
                        confidence=0.75,
                    )
                    region_idx += 1
                    unanchored_regions.append(active_region)

    # Consolidate answer regions
    all_answer_regions: List[AnswerRegion] = []
    seen_ids: Set[str] = set()

    for r in answer_regions + unanchored_regions:
        if r.answer_id in seen_ids:
            continue
        seen_ids.add(r.answer_id)
        all_answer_regions.append(r)

    return StructuredAnswerSheet(
        num_pages=num_pages,
        page_sizes=sizes_list,
        page_analyses=page_analyses,
        question_anchors=all_anchors,
        question_references=all_references,
        answer_regions=answer_regions,
        unanchored_regions=unanchored_regions,
        metadata_blocks=metadata_blocks,
        printed_question_blocks=printed_question_blocks,
        noise_blocks=noise_blocks,
    )


def _bbox_overlaps(a: BBox, b: BBox) -> bool:
    """True if two bboxes share any area."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    return x2 > x1 and y2 > y1


def _region_overlaps_any(vlm_bbox: BBox, vlm_page: int, existing: List[AnswerRegion]) -> bool:
    for reg in existing:
        for r in reg.regions:
            if r.page == vlm_page and _bbox_overlaps(vlm_bbox, r.bbox):
                return True
    return False


def _text_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def _resolve_vlm_text_conflict(ocr_text: str, vlm_text: str, confidence: float) -> Tuple[str, str, bool]:
    """Preserve both OCR and VLM text while selecting the authoritative representation explicitly."""
    ocr_t = (ocr_text or "").strip()
    vlm_t = (vlm_text or "").strip()
    if not vlm_t:
        return ocr_t, "OCR", False
    if not ocr_t:
        return vlm_t, "VLM", False

    similarity = _text_similarity(ocr_t, vlm_t)
    if similarity < 0.25 and confidence >= 0.65:
        return vlm_t, "VLM_REVIEW_REQUIRED", True
    if similarity < 0.45 and confidence >= 0.80:
        return vlm_t, "VLM_REVIEW_REQUIRED", True
    return ocr_t, "OCR", False


def _augment_with_vlm_evidence(
    structured: StructuredAnswerSheet,
    doc_understanding_result: Any,
) -> StructuredAnswerSheet:
    """
    VLM-first precedence:
    1. ground VLM answer regions to actual page geometry
    2. keep OCR only as support text
    3. if the VLM structurally defines a region, it is the authoritative answer candidate
    4. heuristics do not override a grounded VLM answer
    """
    if doc_understanding_result is None:
        return structured

    vlm_regions = [
        r for r in getattr(doc_understanding_result, "regions", [])
        if r.region_type in ("ANSWER_REGION", "HANDWRITING") and getattr(r, "text", "") and str(r.text).strip()
    ]
    if not vlm_regions:
        return structured

    existing_all = structured.answer_regions + structured.unanchored_regions
    next_idx = len(existing_all)

    for vreg in vlm_regions:
        v_bbox = getattr(vreg, "bbox", None)
        v_page = getattr(vreg, "page", 1)
        v_text = (getattr(vreg, "text", "") or "").strip()
        v_conf = max(0.0, min(1.0, float(getattr(vreg, "confidence", 0.0))))
        metadata = getattr(vreg, "metadata", {}) or {}

        anchor = (
            metadata.get("question_number")
            or metadata.get("vlm_display_number")
            or metadata.get("num")
            or metadata.get("answer_to_question_number")
        )
        answer_to = metadata.get("answer_to") or metadata.get("question_anchor")
        grounded_ids = metadata.get("grounded_region_ids") or metadata.get("grounded_ocr_region_ids") or []
        answer_to_conf = float(metadata.get("answer_to_confidence", 0.0) or 0.0)

        overlap_match = None
        for reg in existing_all:
            if reg.pages and v_page not in reg.pages:
                continue
            if any(r.page == v_page and _bbox_overlaps(v_bbox, r.bbox) for r in reg.regions):
                overlap_match = reg
                break

        if overlap_match is not None:
            base = overlap_match
            base.vlm_region_id = getattr(vreg, "region_id", None)
            base.vlm_confidence = max(float(base.vlm_confidence or 0.0), v_conf)
            base.grounded_ocr_region_ids = list(dict.fromkeys((base.grounded_ocr_region_ids or []) + [str(rid) for rid in grounded_ids]))
            base.grounding_status = metadata.get("grounding_status") or base.grounding_status or "GROUNDING_UNKNOWN"
            base.ocr_text = base.ocr_text or base.text or ""
            base.vlm_text = v_text
            base.answer_to = answer_to or base.answer_to
            base.answer_to_question_number = (
                metadata.get("answer_to_question_number")
                or metadata.get("question_number")
                or metadata.get("vlm_display_number")
                or base.answer_to_question_number
            )
            base.answer_to_confidence = max(base.answer_to_confidence, answer_to_conf)
            base.provenance = {
                **(base.provenance or {}),
                "source": "VLM",
                "vlm_region_id": getattr(vreg, "region_id", None),
                "vlm_confidence": v_conf,
                "grounded_ocr_region_ids": base.grounded_ocr_region_ids,
                "grounding_status": base.grounding_status,
                "answer_to": answer_to,
                "question_number": anchor,
            }
            if anchor and not base.question_anchor:
                base.question_anchor = str(anchor).strip()

            selected_text, text_source, review_required = _resolve_vlm_text_conflict(base.ocr_text, base.vlm_text, base.vlm_confidence)
            base.selected_text = selected_text
            base.text_source = text_source
            base.review_required = review_required
            base.needs_review = review_required or base.needs_review
            base.text = selected_text or base.text
            if not base.question_anchor and base.answer_to_question_number:
                base.question_anchor = str(base.answer_to_question_number).strip()
            continue

        anchor_value = str(anchor).strip() if anchor else None
        selected_text, text_source, review_required = _resolve_vlm_text_conflict("", v_text, v_conf)
        new_region = AnswerRegion(
            answer_id=f"vlm_region_{uuid.uuid4().hex[:8]}",
            question_anchor=anchor_value,
            pages=[v_page],
            regions=[Region(page=v_page, bbox=v_bbox)],
            text=selected_text or v_text,
            blocks=[Block(
                id=getattr(vreg, "region_id", f"vlm_{uuid.uuid4().hex[:8]}"),
                text=v_text,
                confidence=max(0.0, min(1.0, float(v_conf))),
                bbox=v_bbox,
                page=v_page,
                source="ocr",
                modality="unknown",
                role="student_answer",
            )],
            reading_order=next_idx,
            confidence=min(0.75, max(0.0, float(v_conf))),
            ocr_text="",
            vlm_text=v_text,
            selected_text=selected_text or v_text,
            text_source=text_source,
            grounding_status=metadata.get("grounding_status") or "UNGROUNDED",
            grounded_ocr_region_ids=[str(rid) for rid in grounded_ids],
            vlm_region_id=getattr(vreg, "region_id", None),
            vlm_confidence=v_conf,
            answer_to=answer_to,
            answer_to_question_number=(metadata.get("answer_to_question_number") or anchor_value),
            answer_to_confidence=answer_to_conf,
            provenance={
                "source": "VLM",
                "vlm_region_id": getattr(vreg, "region_id", None),
                "vlm_confidence": v_conf,
                "grounded_ocr_region_ids": [str(rid) for rid in grounded_ids],
                "grounding_status": metadata.get("grounding_status") or "UNGROUNDED",
                "answer_to": answer_to,
                "question_number": anchor_value,
            },
            review_required=review_required,
            needs_review=review_required,
        )
        next_idx += 1

        if new_region.question_anchor:
            structured.answer_regions.append(new_region)
        else:
            structured.unanchored_regions.append(new_region)
        existing_all.append(new_region)

    return structured


def extract_answers(
    blocks: List[Block],
    metadata_pages: Optional[Set[int]] = None,
    doc_understanding_result: Optional[Any] = None,
) -> List[AnswerCandidate]:
    """
    Backward-compatible entry point for the core processing pipeline.
    Generates AnswerCandidate objects from StructuredAnswerSheet student regions.

    doc_understanding_result (optional): the answer sheet's VLM DocumentUnderstandingResult
    (see pipeline.py STEP 4B). When provided, it is used to fill in answers the regex/anchor
    pass alone could not find or read reliably — see _augment_with_vlm_evidence above.
    """
    num_pages = max((b.page for b in blocks), default=1)
    structured = process_answer_sheet(blocks, num_pages)
    structured = _augment_with_vlm_evidence(structured, doc_understanding_result)

    candidates: List[AnswerCandidate] = []
    for idx, r in enumerate(structured.answer_regions):
        candidates.append(
            AnswerCandidate(
                answer_id=r.answer_id,
                question_number=r.question_anchor,
                text=r.text,
                regions=r.regions,
                order_index=idx,
            )
        )

    # Unanchored student work (no question number could be determined by either pass) is
    # still surfaced as a candidate with question_number=None rather than silently dropped —
    # the mapping/grading stage already has an "unmatched answer" path for exactly this case,
    # and dropping it here means a real student answer would disappear from review entirely
    # (see design principle: preserve evidence, don't hide failures).
    for idx, r in enumerate(structured.unanchored_regions, start=len(candidates)):
        candidates.append(
            AnswerCandidate(
                answer_id=r.answer_id,
                question_number=None,
                text=r.text,
                regions=r.regions,
                order_index=idx,
            )
        )

    return candidates