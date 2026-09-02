"""
Question Extractor — Pure Multimodal VLM Visual Document Understanding.

Understands Question Papers visually and semantically, similar to how a human teacher reads them.
Zero regex heuristics, zero keyword matching, zero OCR line assumptions.
"""
from __future__ import annotations

import io
import base64
import json
import re
from typing import List, Optional, Dict, Any
from PIL import Image

from app.models.schemas import Question, BBox, Block
from app.services.llm_provider import llm_complete_multimodal_with_metadata


def pil_image_to_b64(img: Any) -> str:
    """Encodes PIL Image to JPEG base64 string for VLM inference."""
    buf = io.BytesIO()
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_UUID_RE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.IGNORECASE)


def _sanitize_question_number(raw_num: str, fallback_idx: int) -> str:
    """
    Sanitizes question number while strictly preserving subquestion parts like '1(a)', '5(i)', 'Q3'.
    Rejects UUIDs or empty strings that VLMs may occasionally hallucinate.
    """
    if not raw_num or not raw_num.strip():
        return str(fallback_idx)
    s = raw_num.strip()
    if _UUID_RE.match(s):
        return str(fallback_idx)
    # Strip leading 'Q' or 'q' prefix for the display number (e.g. 'Q1' -> '1', 'Q5(a)' -> '5(a)')
    if s.lower().startswith("q"):
        s = s[1:].strip().lstrip(".")
    # Remove trailing colon/period
    s = s.rstrip(".:-").strip()
    return s if s else str(fallback_idx)


QUESTION_EXTRACTION_PROMPT = """You are an expert examiner and teacher with advanced visual document understanding capabilities.
You are inspecting a QUESTION PAPER image.

Your task: Visually and semantically analyze the layout, typography, numbering, and structure of this document page.
Identify only the ACTUAL ASSESSABLE QUESTIONS that a student is expected to answer.

Return ONLY a valid JSON object in this exact structure (no markdown, no conversational text):
{
  "questions": [
    {
      "number": "1",
      "text": "What is the SI unit of force?",
      "max_marks": 2.0,
      "question_type": "SHORT_ANSWER",
      "options": [],
      "box_2d": [120, 60, 160, 840],
      "parent_question_number": null
    },
    {
      "number": "2(a)",
      "text": "State Ohm's law.",
      "max_marks": 2.0,
      "question_type": "SHORT_ANSWER",
      "options": [],
      "box_2d": [240, 80, 280, 840],
      "parent_question_number": "2"
    },
    {
      "number": "3",
      "text": "Which organelle is known as the powerhouse of the cell?",
      "max_marks": 1.0,
      "question_type": "MCQ",
      "options": ["A. Nucleus", "B. Mitochondria", "C. Ribosome", "D. Golgi apparatus"],
      "box_2d": [400, 60, 480, 840],
      "parent_question_number": null
    }
  ]
}

CRITICAL RULES FOR HUMAN-LIKE UNDERSTANDING:

1. UNDERSTAND DOCUMENT ROLES (IGNORE NON-QUESTIONS):
   - IGNORE document title/header: e.g. "SCIENCE QUESTION PAPER", "ANNUAL EXAMINATION", "Class: 10".
   - IGNORE exam metadata: Time allowed, Maximum marks, Subject code, Date.
   - IGNORE student fields: "Name: _____", "Roll No: _____".
   - IGNORE instructions to candidates: "All questions are compulsory", "Attempt any four questions", "Read instructions carefully".
   - IGNORE section headers: "SECTION A", "PART B", "GROUP I".
   - IGNORE decorative headers, footers, and page numbers.
   - Pure introductory lines (e.g. "2. Answer the following:") are parent instructional headers, NOT standalone questions. Include their subquestions (2(a), 2(b)) with parent_question_number="2".

2. PRESERVE SUBQUESTION HIERARCHY:
   - Understand subquestions like 5(a), 5(b), 5(c) or Q7(i), Q7(ii).
   - PRESERVE the full subpart identifier in "number" (e.g. "5(a)", "5(b)", "7(i)").
   - Set "parent_question_number" to the parent question (e.g. "5" for "5(a)", "7" for "7(i)").
   - DO NOT collapse subquestions into unrelated flat numbers.

3. FIRST-CLASS MCQ SUPPORT:
   - When a question presents multiple choice options (A, B, C, D or (a), (b), (c), (d)), set "question_type": "MCQ".
   - Populate "options" with the option strings, preserving their labels (e.g. ["A. Earth", "B. Mars", "C. Jupiter", "D. Venus"]).
   - Options arranged horizontally, in columns, or in tables must be structurally attached to the question.

4. SUPPORT ALL QUESTION TYPES:
   - "question_type" can be: "MCQ", "SHORT_ANSWER", "LONG_ANSWER", "NUMERICAL", "TABLE", "DIAGRAM".
   - If a question includes a diagram, graph, or table, include the description or prompt in "text" and set question_type accordingly.

5. MARKS:
   - Where marks are indicated (e.g. "[2 Marks]", "(5)", "[10]"), extract the numeric value into "max_marks". Default to 2.0 if not specified.
   - Do not confuse marks with question numbers (e.g. "5. Define energy. (3)" means question 5, marks 3.0).

6. BOUNDING BOXES FOR UI HIGHLIGHTING:
   - Provide "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000 integers enclosing the question on the page.
   - [0, 0] is TOP-LEFT, [1000, 1000] is BOTTOM-RIGHT of the image.
"""


async def extract_questions_vlm(qp_images_dict: Dict[int, Any]) -> List[Question]:
    """
    100% Multimodal VLM Visual Question Paper Extractor.
    Extracts structured assessable questions directly from question paper page images.
    """
    all_questions: List[Question] = []
    order_idx = 0

    for page_num in sorted(qp_images_dict.keys()):
        img = qp_images_dict[page_num]
        b64_img = pil_image_to_b64(img)

        img_w = getattr(img, "width", 1000) if hasattr(img, "width") else 1000
        img_h = getattr(img, "height", 1400) if hasattr(img, "height") else 1400

        try:
            raw_res, meta = await llm_complete_multimodal_with_metadata(
                prompt=QUESTION_EXTRACTION_PROMPT,
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
                raw_num = str(q_dict.get("number", "")).strip()
                clean_num = _sanitize_question_number(raw_num, order_idx + 1)
                q_text = str(q_dict.get("text", "")).strip()
                if not q_text:
                    continue

                try:
                    m_marks = float(q_dict.get("max_marks", 2.0) or 2.0)
                except (TypeError, ValueError):
                    m_marks = 2.0

                q_type_raw = str(q_dict.get("question_type", "SHORT_ANSWER")).upper().strip()
                q_type = q_type_raw if q_type_raw in (
                    "MCQ", "SHORT_ANSWER", "LONG_ANSWER", "NUMERICAL", "TABLE", "DIAGRAM", "SUBQUESTION"
                ) else "SHORT_ANSWER"

                opts = q_dict.get("options", []) or []
                if not isinstance(opts, list):
                    opts = [str(opts)]
                clean_opts = [str(o).strip() for o in opts if str(o).strip()]

                # Bounding box for exact frontend highlighting
                box_raw = q_dict.get("box_2d") or q_dict.get("bbox")
                q_bbox: Optional[BBox] = None
                if isinstance(box_raw, list) and len(box_raw) > 0:
                    flat_box = box_raw[0] if isinstance(box_raw[0], list) else box_raw
                    if len(flat_box) == 4:
                        try:
                            ymin, xmin, ymax, xmax = [float(v) for v in flat_box]
                            by = (ymin / 1000.0) * img_h
                            bx = (xmin / 1000.0) * img_w
                            bh = max(5.0, ((ymax - ymin) / 1000.0) * img_h)
                            bw = max(10.0, ((xmax - xmin) / 1000.0) * img_w)
                            q_bbox = BBox(x=round(bx, 1), y=round(by, 1), width=round(bw, 1), height=round(bh, 1))
                        except (TypeError, ValueError):
                            q_bbox = None
                elif isinstance(box_raw, dict):
                    try:
                        bx = float(box_raw.get("x", 0) or 0)
                        by = float(box_raw.get("y", 0) or 0)
                        bw = float(box_raw.get("width", 0) or 0)
                        bh = float(box_raw.get("height", 0) or 0)
                        if bw > 0 and bh > 0:
                            bx = max(0.0, min(bx, img_w - 1))
                            by = max(0.0, min(by, img_h - 1))
                            bw = max(1.0, min(bw, img_w - bx))
                            bh = max(1.0, min(bh, img_h - by))
                            q_bbox = BBox(x=round(bx, 1), y=round(by, 1), width=round(bw, 1), height=round(bh, 1))
                    except (TypeError, ValueError):
                        q_bbox = None

                # Subquestion parent linkage
                parent_q_raw = q_dict.get("parent_question_number") or None
                parent_q_id: Optional[str] = None
                if parent_q_raw:
                    clean_parent = _sanitize_question_number(str(parent_q_raw), 0)
                    if clean_parent and clean_parent != "0":
                        parent_q_id = f"Q{clean_parent}"

                q_obj = Question(
                    id=f"Q{clean_num}",
                    number=clean_num,
                    text=q_text,
                    page=page_num,
                    bbox=q_bbox,
                    order_index=order_idx,
                    question_type=q_type,  # type: ignore
                    options=clean_opts,
                    parent_question_id=parent_q_id,
                    max_marks=m_marks,
                    extraction_confidence=1.0 if meta.get("vlm_result") == "SUCCESS" else 0.8,
                )
                all_questions.append(q_obj)
                order_idx += 1
                print(f"[VLMQuestionExtractor] Extracted Q{clean_num} (type={q_type}, marks={m_marks}, parent={parent_q_id}) on page {page_num}: '{q_text[:50]}'")

        except Exception as e:
            print(f"[VLMQuestionExtractor] Error extracting questions on page {page_num}: {e}")

    # Natural sort: page -> primary question number -> subpart
    def _sort_key(q: Question):
        m = re.match(r"^\s*(\d{1,3})", q.number)
        val = int(m.group(1)) if m else 999
        sub_m = re.search(r"[\(\[]([a-z]{1,2}|[ivxl]{1,4})[\)\]]", q.number, re.IGNORECASE)
        sub_val = ord(sub_m.group(1).lower()[0]) if sub_m else -1
        return (q.page, val, sub_val, q.order_index)

    all_questions.sort(key=_sort_key)
    for idx, q in enumerate(all_questions):
        q.order_index = idx

    print(f"[VLMQuestionExtractor] Total: {len(all_questions)} questions extracted.")
    return all_questions


async def extract_questions(
    blocks: Optional[List[Block]] = None,
    qp_images: Optional[Dict[int, Any]] = None,
    **kwargs: Any,
) -> List[Question]:
    """
    Unified extraction entry point.
    Runs 100% VLM visual extraction on question paper images.
    """
    if qp_images:
        return await extract_questions_vlm(qp_images)
    return []
