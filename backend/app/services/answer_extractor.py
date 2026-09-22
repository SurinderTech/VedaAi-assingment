"""
Answer Extractor — Pure Multimodal VLM Visual Document Understanding.

Understands Student Answer Sheets visually and semantically, similar to how a human examiner reads them.
Zero regex heuristics, zero column clustering assumptions, zero OCR anchor rules.
Handles handwritten answers, printed answers, MCQ selections (circled, ticked, letters, tables),
and subquestion answers, while preserving exact pixel bounding boxes for UI highlighting.
"""
from __future__ import annotations

import io
import asyncio
import base64
import json
import re
from typing import List, Optional, Dict, Any
from PIL import Image

from app.models.schemas import AnswerCandidate, Region, BBox, Block
from app.services.llm_provider import llm_complete_multimodal_with_metadata


def pil_image_to_b64(img: Any) -> str:
    """Encodes PIL Image to JPEG base64 string for VLM inference."""
    buf = io.BytesIO()
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=75)  # 75 = faster upload/VLM call, still readable
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_UUID_RE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.IGNORECASE)


def _clean_question_reference(raw_num: str) -> str:
    """
    Cleans leading prefixes ('Ans', 'Q.') while preserving the exact
    question reference identified by the VLM ('1', '2(a)', '5(i)').
    Zero regex anchor guessing or digit collapsing.
    """
    if not raw_num or not raw_num.strip():
        return ""
    s = raw_num.strip()
    if _UUID_RE.match(s):
        return ""
    for pfx in ("ans.", "ans:", "ans", "answer.", "answer:", "answer", "q.", "q:", "q"):
        if s.lower().startswith(pfx):
            s = s[len(pfx):].strip()
            break
    s = s.strip(".:- ")
    if s.startswith("(") and s.endswith(")") and "(" not in s[1:-1]:
        s = s[1:-1].strip()
    return s


ANSWER_EXTRACTION_PROMPT = """You are an expert exam evaluator and human teacher with computer vision capabilities.
You are inspecting a STUDENT ANSWER SHEET image.

The answers written on this page may be:
- Handwritten answers (varying handwriting styles, neat or messy, cursive, printed)
- Digital / printed answers
- MCQ answer selections:
  * An option letter written directly: e.g. "(D)", "B", "1. (D) Combustion"
  * Circled option letter or tick mark beside an option
  * Marked bubbles or boxes
  * A table of answers: e.g. a table with columns "Q.No | Answer" where each cell is an MCQ choice
- Subquestion answers: e.g. "Ans 1(a)", "2(b)", "5(i)"
- Answers written below questions, in margins, in answer boxes, or in dedicated sections

Your task: Visually locate, transcribe, and extract EVERY student answer on this page.

Return ONLY a valid JSON object in this exact structure (no markdown, no explanation):
{
  "answers": [
    {
      "question_number": "1",
      "answer_text": "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
      "box_2d": [140, 60, 210, 840]
    },
    {
      "question_number": "2(a)",
      "answer_text": "Ohm's law states that V = I * R under constant temperature.",
      "box_2d": [240, 60, 300, 840]
    },
    {
      "question_number": "3",
      "answer_text": "(B) Mitochondria",
      "box_2d": [460, 60, 500, 460]
    }
  ]
}

CRITICAL RULES FOR HUMAN-LIKE VISION:

1. QUESTION NUMBER / ANSWER RELATIONSHIP:
   - Identify which question number the answer belongs to ONLY from visible anchors, row labels, or written references.
   - For subquestions, PRESERVE subparts: e.g. "1(a)", "1(b)", "5(i)", "5(ii)". DO NOT strip the subpart letter!
   - Clean prefix: "Ans 1." -> "1", "Q2(a)" -> "2(a)", "(3)" -> "3".
   - CRITICAL: If the student DID NOT write a question number or reference label beside an answer, set "question_number": null. DO NOT guess, count items, or invent question numbers.

2. FAITHFUL TRANSCRIPTION (HANDWRITING & PRINT):
   - Read what the student actually wrote visually.
   - For MCQs: capture the chosen option letter and any accompanying text, e.g. "(B) Mitochondria" or "B".
   - If the student crossed out text, transcribe their final intended answer.

3. BOUNDING BOXES FOR UI HIGHLIGHTING:
   - Provide "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000 integers enclosing the entire answer block (and question number if attached).
   - [0, 0] is TOP-LEFT, [1000, 1000] is BOTTOM-RIGHT of the image.
   - The bounding box is used by the frontend to visually highlight the answer evidence to the teacher.

4. DO NOT EXTRACT NON-ANSWERS:
   - Do NOT treat student identity fields ("Name: Alex", "Roll: 101"), exam headers, instructions, or page numbers as answers.
   - If a question on the paper is NOT answered on this page, do NOT invent an answer for it.
"""


async def _extract_answers_from_page(
    page_num: int,
    img: Any,
) -> List[AnswerCandidate]:
    """
    Extracts answers from a single answer sheet page image via VLM.
    Designed to be called concurrently across all pages using asyncio.gather.
    """
    b64_img = pil_image_to_b64(img)
    img_w = getattr(img, "width", 1000) if hasattr(img, "width") else 1000
    img_h = getattr(img, "height", 1400) if hasattr(img, "height") else 1400
    page_candidates: List[AnswerCandidate] = []

    try:
        raw_res, meta = await llm_complete_multimodal_with_metadata(
            prompt=ANSWER_EXTRACTION_PROMPT,
            image_b64=b64_img,
            mime_type="image/jpeg",
            purpose=f"vlm_answer_extraction_p{page_num}",
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
        raw_ans = data.get("answers", []) if isinstance(data, dict) else []

        for idx, a_dict in enumerate(raw_ans):
            q_num_raw = str(a_dict.get("question_number", "")).strip()
            clean_q_num = _clean_question_reference(q_num_raw)
            a_text = str(a_dict.get("answer_text", "")).strip()

            if not a_text and not clean_q_num:
                continue

            box_raw = a_dict.get("box_2d") or a_dict.get("bbox")
            raw_x, raw_y, raw_w, raw_h = 0.0, 0.0, img_w * 0.7, 50.0

            if isinstance(box_raw, list) and len(box_raw) > 0:
                flat_box = box_raw[0] if isinstance(box_raw[0], list) else box_raw
                if len(flat_box) == 4:
                    try:
                        ymin, xmin, ymax, xmax = [float(v) for v in flat_box]
                        raw_y = (ymin / 1000.0) * img_h
                        raw_x = (xmin / 1000.0) * img_w
                        raw_h = max(10.0, ((ymax - ymin) / 1000.0) * img_h)
                        raw_w = max(10.0, ((xmax - xmin) / 1000.0) * img_w)
                    except (ValueError, TypeError):
                        pass
            elif isinstance(box_raw, dict):
                vlm_page_w = float(data.get("page_width", img_w)) if isinstance(data, dict) else img_w
                vlm_page_h = float(data.get("page_height", img_h)) if isinstance(data, dict) else img_h
                scale_x = img_w / vlm_page_w if vlm_page_w > 0 else 1.0
                scale_y = img_h / vlm_page_h if vlm_page_h > 0 else 1.0
                raw_x = float(box_raw.get("x", 0) or 0) * scale_x
                raw_y = float(box_raw.get("y", 0) or 0) * scale_y
                raw_w = float(box_raw.get("width", img_w * 0.7) or (img_w * 0.7)) * scale_x
                raw_h = float(box_raw.get("height", 50) or 50) * scale_y

            clamped_x = max(0.0, min(raw_x, img_w - 1))
            clamped_y = max(0.0, min(raw_y, img_h - 1))
            clamped_w = max(10.0, min(raw_w, img_w - clamped_x))
            clamped_h = max(10.0, min(raw_h, img_h - clamped_y))

            region = Region(
                page=page_num,
                bbox=BBox(
                    x=round(clamped_x, 1),
                    y=round(clamped_y, 1),
                    width=round(clamped_w, 1),
                    height=round(clamped_h, 1),
                ),
            )

            cand = AnswerCandidate(
                answer_id=f"ans_vlm_p{page_num}_{idx}",
                question_number=f"Q{clean_q_num}" if clean_q_num else None,
                text=a_text,
                regions=[region],
                order_index=idx,  # Temporary; re-indexed globally after gather
            )
            page_candidates.append(cand)
            print(f"[VLMAnswerExtractor] Extracted Answer for Q{clean_q_num} on page {page_num}: '{a_text[:45]}' bbox=({clamped_x:.0f},{clamped_y:.0f},{clamped_w:.0f},{clamped_h:.0f})")

    except Exception as e:
        print(f"[VLMAnswerExtractor] Error extracting answers on page {page_num}: {e}")

    return page_candidates


async def extract_answers_vlm(as_images_dict: Dict[int, Any]) -> List[AnswerCandidate]:
    """
    100% Multimodal VLM Visual Answer Sheet Extractor.
    Transcribes student answers directly from answer sheet images.
    Preserves exact bounding boxes and subquestion links.

    ALL pages are processed CONCURRENTLY using asyncio.gather — dramatically reducing
    extraction time for long answer sheets.
    """
    page_nums = sorted(as_images_dict.keys())
    print(f"[VLMAnswerExtractor] Processing {len(page_nums)} answer sheet page(s) in parallel...")

    tasks = [
        _extract_answers_from_page(page_num, as_images_dict[page_num])
        for page_num in page_nums
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: List[AnswerCandidate] = []
    order_idx = 0
    for page_num, page_result in zip(page_nums, results):
        if isinstance(page_result, Exception):
            print(f"[VLMAnswerExtractor] Page {page_num} failed: {page_result}")
            continue
        for cand in page_result:
            cand.order_index = order_idx
            candidates.append(cand)
            order_idx += 1

    print(f"[VLMAnswerExtractor] Total: {len(candidates)} answer candidates extracted from {len(page_nums)} page(s).")
    return candidates


def extract_answers(
    blocks: Optional[List[Block]] = None,
    as_images: Optional[Dict[int, Any]] = None,
    **kwargs: Any,
) -> List[AnswerCandidate]:
    """
    Unified answer extraction entry point.
    Runs 100% VLM visual extraction on answer sheet images.
    """
    if as_images:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                # In async context, caller should use extract_answers_vlm directly
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(extract_answers_vlm(as_images))
            else:
                return asyncio.run(extract_answers_vlm(as_images))
        except RuntimeError:
            return asyncio.run(extract_answers_vlm(as_images))
    return []