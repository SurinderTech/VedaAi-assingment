"""
Question Requirement Analyzer & Structural Marks Resolver.

Converts an extracted Question into a QuestionRequirementSpec detailing:
- Expected answer type taxonomy (definition, mcq, one_word, numerical, diagram, code, long_conceptual, etc.)
- Resolved maximum marks (derived from question text, section headers, or structural context)
- Required concepts, operations, and visual/numerical/code flags.
"""
from __future__ import annotations
import re
from typing import Optional, List, Tuple
from app.models.schemas import Question, QuestionRequirementSpec, QuestionType


def resolve_marks_from_context(question: Question) -> Tuple[float, str]:
    """
    Dynamically determines expected max marks from explicit question text, section instructions,
    or structural context without hardcoding specific question numbers.
    """
    text = question.text or ""
    sec = question.section or ""
    
    # 1. Direct mark patterns in question text e.g. "[10 Marks]", "(2 marks)", "10M", "[5]"
    m_direct = re.search(r"\[?\b(\d{1,2}(?:\.\d)?)\s*(?:marks?|m|pts?|points?)\b\]?", text, re.IGNORECASE)
    if m_direct:
        return float(m_direct.group(1)), "high"
    
    # 2. Section context patterns e.g. "SECTION-A: TEN questions TWO marks each"
    full_context = f"{sec} {text}"
    m_sec_each = re.search(r"(?:each|per)\s+(?:question|item)?\s*(?:carries|is|of)?\s*(\d{1,2})\s*marks?", full_context, re.IGNORECASE)
    if m_sec_each:
        return float(m_sec_each.group(1)), "high"
        
    m_word_marks = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*marks?\b", full_context, re.IGNORECASE)
    word_map = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0}
    if m_word_marks and m_word_marks.group(1).lower() in word_map:
        return word_map[m_word_marks.group(1).lower()], "high"

    # Section heuristic mapping
    sec_lower = sec.lower()
    if "section-a" in sec_lower or "part-a" in sec_lower or "part a" in sec_lower:
        return 2.0, "medium"
    if "section-b" in sec_lower or "part-b" in sec_lower or "part b" in sec_lower:
        return 5.0, "medium"
    if "section-c" in sec_lower or "part-c" in sec_lower or "part c" in sec_lower:
        return 10.0, "medium"
        
    return 2.0, "medium"


def analyze_question_requirements(question: Question) -> QuestionRequirementSpec:
    """
    Analyzes question prompt, options, and section metadata to produce an evaluation spec.
    """
    txt = (question.text or "").strip()
    txt_lower = txt.lower()
    options = question.options or []
    
    max_marks, confidence = resolve_marks_from_context(question)
    
    # Flags
    has_diagram = bool(re.search(r"\b(diagram|sketch|draw|figure|flowchart|architecture|visual)\b", txt_lower))
    has_math = bool(re.search(r"\b(calculate|compute|solve|derive|equation|formula|evaluate|find the value|matrix|integral)\b", txt_lower) or re.search(r"[=\+\-\*/\^√∫]", txt))
    has_code = bool(re.search(r"\b(code|program|python|function|algorithm|implement|script|class|java|c\+\+)\b", txt_lower))
    
    # Classification
    answer_type: QuestionType = "unknown"
    
    if len(options) >= 2 or re.search(r"\([A-D]\)", txt) or re.search(r"^\s*\(?[A-D]\)?\s*[\.\-]", txt):
        answer_type = "mcq"
    elif re.search(r"^\s*(name|state|what is the output of|which|who|where|unit of)\b", txt_lower) and len(txt.split()) <= 12 and not has_math:
        answer_type = "one_word"
    elif re.search(r"\b(define|meaning of|what is|what do you understand by)\b", txt_lower) and len(txt.split()) <= 15:
        answer_type = "definition"
    elif has_code or "algorithm" in txt_lower:
        answer_type = "code" if has_code else "algorithm"
    elif has_math:
        if "derive" in txt_lower or "proof" in txt_lower:
            answer_type = "mathematical_derivation"
        else:
            answer_type = "numerical"
    elif has_diagram and ("explain" in txt_lower or "describe" in txt_lower):
        answer_type = "diagram_explanation"
    elif has_diagram:
        answer_type = "diagram"
    elif re.search(r"\b(differentiate|compare|difference between|vs|versus)\b", txt_lower):
        answer_type = "comparison"
    elif re.search(r"\b(steps|process|procedure|working of|how does)\b", txt_lower):
        answer_type = "process"
    elif max_marks >= 7.0 or len(txt.split()) > 25 or "with example" in txt_lower or "in detail" in txt_lower:
        answer_type = "long_conceptual"
    elif max_marks <= 2.0 or len(txt.split()) <= 15:
        answer_type = "short_conceptual"
    else:
        answer_type = "explanation"

    # Extract core required concepts (noun phrases / key terms)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", txt)
    stop_words = {
        "what", "when", "where", "which", "who", "why", "how", "define", "explain", "differentiate",
        "between", "mention", "major", "functions", "given", "write", "state", "using", "with",
        "suitable", "question", "marks", "answer", "find", "show", "calculate", "identify", "describe",
        "detail", "help", "example"
    }
    concepts = list(dict.fromkeys([w.lower() for w in words if w.lower() not in stop_words]))
    
    # Requirement descriptions
    criteria_descs = []
    if answer_type == "mcq":
        criteria_descs.append("Selects correct option / choice")
    elif answer_type == "one_word":
        criteria_descs.append("Provides accurate factual name or output")
    elif answer_type == "definition":
        criteria_descs.append(f"Provides accurate definition of {', '.join(concepts[:2]) if concepts else 'target topic'}")
    else:
        for c in concepts[:4]:
            criteria_descs.append(f"Addresses concept: {c}")
            
    if has_diagram:
        criteria_descs.append("Includes required diagram or visual representation")
    if has_math:
        criteria_descs.append("Provides mathematical steps and correct calculation")

    return QuestionRequirementSpec(
        expected_answer_type=answer_type,
        max_marks=max_marks,
        marks_confidence=confidence,
        required_concepts=concepts,
        required_operations=["calculate"] if has_math else (["draw"] if has_diagram else ["explain"]),
        has_diagram_requirement=has_diagram,
        has_numerical_requirement=has_math,
        has_code_requirement=has_code,
        evaluation_criteria_descriptions=criteria_descs,
    )
