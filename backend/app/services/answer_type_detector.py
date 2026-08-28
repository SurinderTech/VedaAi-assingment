"""
Answer Content & Modality Detector.

Analyzes student answer text, region structure, and bounding boxes to determine content type
(e.g., short_text, long_text, number, formula, mathematical_work, code, diagram, mcq_selection, mixed, visual_only).
Runs independently of OCR modality (printed vs handwritten).
"""
from __future__ import annotations
import re
from typing import Optional
from app.models.schemas import MappedAnswer, AnswerContentType


def detect_answer_content_type(mapped_answer: MappedAnswer, raw_text: Optional[str] = None) -> AnswerContentType:
    """
    Determines answer content type from text content, mathematical symbols, code patterns, or region bounding boxes.
    """
    text = (raw_text or mapped_answer.text or "").strip()
    word_count = len(text.split())
    
    if not text:
        if mapped_answer.regions:
            return "visual_only"
        return "unknown"
        
    # MCQ option selection e.g. "A", "(B)", "Option C", "d"
    if re.match(r"^\s*\(?[A-Da-d]\)?\s*$", text) or re.match(r"^\s*(?:option|ans(?:wer)?)\s*[:\.\-]?\s*\(?[A-Da-d]\)?\s*$", text, re.IGNORECASE):
        return "mcq_selection"
        
    # Standalone number e.g. "0", "-5", "3.14159", "42"
    if re.match(r"^\s*[\+\-]?\d+(?:\.\d+)?\s*$", text):
        return "number"
        
    # Code patterns
    if re.search(r"\b(def |class |import |for |while |return |public static|if __name__|std::)\b", text) or re.search(r"[{};]\s*$", text, re.MULTILINE):
        return "code"
        
    # Math / Formula work
    has_math_symbols = bool(re.search(r"[=\+\-\*/\^√∫Σλπθ∂]|\\frac|\\sqrt", text))
    if has_math_symbols:
        if word_count <= 5:
            return "formula"
        return "mathematical_work"
        
    # Text length categories
    if word_count <= 4:
        return "short_text"
    elif word_count >= 40:
        return "long_text"
    else:
        return "text"
