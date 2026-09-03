"""
Intelligent Multiple Choice Question (MCQ) Evaluator & Answer Key Engine.

Features:
1. Robust Student Option Parsing:
   Extracts option letters (A, B, C, D) from real-world handwritten/OCR answers:
   - "Q21. (A) A convex lens has 4 dioptre power..."
   - "(A) is less than one"
   - "A. option text"
   - "Option C"
   - "Ans: (B)"
   - "fruits peels, cake and lime juice" (semantic option text matching when letter is omitted)
2. Batch Answer Key Generation:
   Solves question paper MCQs using a single fast LLM call.
3. Deterministic Evidence Generation:
   Awards full marks with high confidence when student option matches correct option,
   and 0 marks with clear diagnostic feedback when mismatched.
"""
from __future__ import annotations

import re
import asyncio
from typing import List, Optional, Tuple, Dict, Any

from app.models.schemas import Question, MappedAnswer, Rubric, CriterionEvidence
from app.services.llm_provider import llm_complete_json
from app.services.embedding_service import similarity_matrix


def _clean_option_label(label_or_text: str) -> str:
    """Extracts clean uppercase letter A-D from an option string or label."""
    m = re.search(r"\b([A-Da-d])\b", label_or_text)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\(?([A-Da-d])\)?", label_or_text)
    if m2:
        return m2.group(1).upper()
    return ""


def _strip_option_prefix(text: str) -> str:
    """Strips leading question number or option label from text."""
    s = re.sub(r"^\s*(?:(?:Q|q)?\d+[\.\)\-]?\s*)", "", text).strip()
    s = re.sub(r"^\s*\(?[A-Da-d]\)?(?:[\.\:\-\)]|\s+)", "", s).strip()
    return s


def extract_student_option_choice(question: Question, answer_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts (selected_option_letter, selected_option_text) from student's answer text.
    Handles multiple variations seen on student answer sheets.
    """
    text = (answer_text or "").strip()
    if not text:
        return None, ""

    # Pattern 1: Leading question number + option letter, e.g. "Q21. (A) ...", "21. (A) ...", "Q21(A)"
    m1 = re.match(
        r"^\s*(?:(?:Q|q)?\d+[\.\)\-]?\s*)?\(?([A-Da-d])\)?(?:[\.\:\-\)]|\s+|$)(.*)",
        text,
        re.DOTALL
    )
    if m1:
        letter = m1.group(1).upper()
        rem_text = m1.group(2).strip()
        return letter, rem_text or text

    # Pattern 2: "Option A", "Ans: A", "Answer - (B)"
    m2 = re.search(
        r"\b(?:option|ans(?:wer)?|choice)\s*[:\.\-]?\s*\(?([A-Da-d])\)?\b(.*)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if m2:
        letter = m2.group(1).upper()
        rem_text = m2.group(2).strip()
        return letter, rem_text or text

    # Pattern 3: Lone letter e.g. "A", "(B)", "c."
    m3 = re.match(r"^\s*\(?([A-Da-d])\)?[\.\:\-]?\s*$", text)
    if m3:
        letter = m3.group(1).upper()
        # Find corresponding option text from question.options if available
        opt_text = ""
        for opt in question.options:
            if _clean_option_label(opt) == letter:
                opt_text = _strip_option_prefix(opt)
                break
        return letter, opt_text or letter

    # Pattern 4: Student omitted the letter and only wrote the option's text!
    # E.g. student wrote "fruits peels, cake and lime juice" for Option C.
    if question.options:
        cleaned_options = []
        for opt in question.options:
            opt_letter = _clean_option_label(opt)
            opt_body = _strip_option_prefix(opt).strip().lower()
            cleaned_options.append((opt_letter, opt_body))

        student_lower = text.lower()
        # Direct substring match
        for opt_letter, opt_body in cleaned_options:
            if opt_body and len(opt_body) > 3 and (opt_body in student_lower or student_lower in opt_body):
                return opt_letter, text

        # TF-IDF similarity match
        bodies = [cb[1] for cb in cleaned_options if cb[1]]
        if bodies:
            try:
                sims = similarity_matrix(bodies, [student_lower])
                best_idx = int(sims.argmax())
                if sims[best_idx, 0] >= 0.40:
                    matched_letter = cleaned_options[best_idx][0]
                    if matched_letter:
                        return matched_letter, text
            except Exception:
                pass

    # Pattern 5: Any isolated (A)-(D) in the first 25 characters
    m5 = re.search(r"\(?([A-Da-d])\)?", text[:25])
    if m5:
        return m5.group(1).upper(), text

    return None, text


async def resolve_answer_key_batch(questions: List[Question]) -> None:
    """
    Batch-solves exam questions using LLM to generate an authoritative answer key
    for all MCQs and factual questions before grading begins.
    Populates question.correct_option and question.correct_answer in place.
    """
    mcqs_to_solve = []
    for q in questions:
        has_opts = bool(q.options and len(q.options) >= 2)
        is_mcq = q.question_type == "MCQ" or has_opts or "(A)" in (q.text or "")
        if is_mcq and not q.correct_option:
            mcqs_to_solve.append(q)

    if not mcqs_to_solve:
        return

    # Construct concise batch payload
    items = []
    for q in mcqs_to_solve:
        items.append({
            "number": str(q.number),
            "text": q.text,
            "options": q.options if q.options else [],
        })

    prompt = (
        "You are an expert examiner and teacher. Solve the following multiple choice exam questions.\n"
        "For each question, determine the single correct option letter (A, B, C, or D), the option text, "
        "and a brief 1-sentence explanation.\n\n"
        f"Questions:\n{items}\n\n"
        "Return ONLY a JSON object in this exact format:\n"
        "{\n"
        '  "answers": [\n'
        '    {"number": "20", "correct_option": "C", "correct_text": "...", "explanation": "..."},\n'
        '    {"number": "21", "correct_option": "A", "correct_text": "...", "explanation": "..."}\n'
        "  ]\n"
        "}"
    )

    try:
        data = await asyncio.wait_for(llm_complete_json(prompt), timeout=12.0)
        if isinstance(data, dict) and "answers" in data and isinstance(data["answers"], list):
            ans_map = {}
            for ans in data["answers"]:
                if isinstance(ans, dict):
                    num_key = str(ans.get("number", "")).strip().lstrip("Q").lstrip("q").strip()
                    corr_opt = str(ans.get("correct_option", "")).strip().upper()
                    if corr_opt:
                        ans_map[num_key] = {
                            "correct_option": corr_opt,
                            "correct_answer": str(ans.get("correct_text", "")).strip(),
                            "explanation": str(ans.get("explanation", "")).strip(),
                        }

            for q in mcqs_to_solve:
                q_num_key = str(q.number).strip().lstrip("Q").lstrip("q").strip()
                if q_num_key in ans_map:
                    q.correct_option = ans_map[q_num_key]["correct_option"]
                    q.correct_answer = ans_map[q_num_key]["correct_answer"]
                    q.explanation = ans_map[q_num_key]["explanation"]
                    print(f"[AnswerKey] Resolved Q{q.number}: Option ({q.correct_option})")
    except Exception as e:
        print(f"[AnswerKey] Batch resolution skipped/failed: {e}")


def evaluate_mcq_evidence(
    question: Question,
    mapped_answer: MappedAnswer,
    rubric: Rubric,
) -> Optional[List[CriterionEvidence]]:
    """
    Evaluates MCQ student answer deterministically when correct_option is known.
    Returns CriterionEvidence list if successfully evaluated, or None if fallback needed.
    """
    has_opts = bool(question.options and len(question.options) >= 2)
    is_mcq = question.question_type == "MCQ" or has_opts or "(A)" in (question.text or "")
    if not is_mcq:
        return None

    total_marks = rubric.total_max_marks or 1.0
    text = (mapped_answer.text or "").strip()

    # Handle unanswered
    if mapped_answer.status == "unanswered" or not text and not mapped_answer.regions:
        return [
            CriterionEvidence(
                criterion_id="c1",
                description="Selects the correct option choice",
                status="missing",
                evidence_text=None,
                confidence=0.98,
                awarded_marks=0.0,
                max_marks=total_marks,
                notes="No student answer provided",
                provenance="local",
            )
        ]

    # Extract student selection
    student_letter, student_text = extract_student_option_choice(question, text)

    # If correct_option is known:
    if question.correct_option:
        correct_letter = question.correct_option.strip().upper()
        if student_letter:
            is_correct = (student_letter == correct_letter)
            if is_correct:
                opt_display = student_text if student_text != student_letter else ""
                disp_str = f"({student_letter}) {opt_display}".strip()
                return [
                    CriterionEvidence(
                        criterion_id="c1",
                        description=f"Selects correct option ({correct_letter})",
                        status="present",
                        evidence_text=f"Student correctly selected {disp_str}",
                        confidence=0.98,
                        awarded_marks=total_marks,
                        max_marks=total_marks,
                        notes=f"✓ Correct option ({student_letter}) selected. {question.explanation or ''}".strip(),
                        provenance="local",
                    )
                ]
            else:
                return [
                    CriterionEvidence(
                        criterion_id="c1",
                        description=f"Selects correct option ({correct_letter})",
                        status="missing",
                        evidence_text=f"Student selected ({student_letter}), but correct option is ({correct_letter})",
                        confidence=0.98,
                        awarded_marks=0.0,
                        max_marks=total_marks,
                        notes=f"✗ Incorrect option ({student_letter}) selected. Correct answer is ({correct_letter}). {question.explanation or ''}".strip(),
                        provenance="local",
                    )
                ]
        else:
            # Student wrote some text but option letter could not be parsed
            return [
                CriterionEvidence(
                    criterion_id="c1",
                    description=f"Selects correct option ({correct_letter})",
                    status="uncertain",
                    evidence_text=text[:100],
                    confidence=0.50,
                    awarded_marks=0.0,
                    max_marks=total_marks,
                    notes=f"Could not unambiguously resolve selected option choice from text: '{text[:60]}'",
                    provenance="local",
                )
            ]

    # If correct_option is NOT yet known, return None to escalate to LLM
    return None
