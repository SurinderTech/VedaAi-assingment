"""
Code Answer Evaluator.

Performs static syntax analysis, structural parsing, and logic checks for code responses
without relying on semantic similarity alone.
"""
from __future__ import annotations
import ast
import re
from typing import Dict, Any
from app.models.schemas import Question, MappedAnswer


def evaluate_code_answer(question: Question, mapped_answer: MappedAnswer) -> Dict[str, Any]:
    """
    Evaluates code answers using static syntax verification and logical structure analysis.
    """
    code_text = (mapped_answer.text or "").strip()
    if not code_text:
        return {
            "code_score": 0.0,
            "is_valid_syntax": False,
            "has_logic": False,
            "confidence": 0.90,
            "notes": "No code provided",
        }

    is_valid_syntax = False
    # Attempt Python AST parse
    try:
        ast.parse(code_text)
        is_valid_syntax = True
    except Exception:
        # Generic syntax structure check e.g. function defs, keywords, braces
        is_valid_syntax = bool(re.search(r"\b(def|class|for|while|if|return|public|void)\b", code_text))

    has_logic = bool(re.search(r"\b(return|print|output|yield|def|class)\b", code_text))
    
    score = 0.50
    if is_valid_syntax and has_logic:
        score = 0.95
    elif is_valid_syntax or has_logic:
        score = 0.70

    return {
        "code_score": score,
        "is_valid_syntax": is_valid_syntax,
        "has_logic": has_logic,
        "confidence": 0.85 if is_valid_syntax else 0.65,
        "notes": f"Code static check: syntax={is_valid_syntax}, logic={has_logic}",
    }
