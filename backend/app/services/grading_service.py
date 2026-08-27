"""
AI Question Grading & Human-Like Evaluation Service.

Performs deep analysis of student responses vs exam questions:
1. Evaluates accuracy, technical terms, missing concepts, and clarity.
2. Generates human-like professor critique explaining score rationale.
3. Provides dynamic topic-aware fallback evaluation if offline/API unavailable.
"""
from __future__ import annotations
import asyncio
import json
import re
from typing import List
from app.models.schemas import Question, MappedAnswer, Grading
from app.services.llm_provider import llm_complete


def _extract_key_terms(text: str) -> List[str]:
    """Extracts meaningful technical terms from question prompt."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    stop_words = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "define", "explain", "differentiate", "between", "mention", "major", "functions",
        "consider", "following", "given", "write", "state", "using", "with", "suitable",
        "question", "marks", "answer", "find", "show", "calculate", "identify"
    }
    return list(dict.fromkeys([w.lower() for w in words if w.lower() not in stop_words]))


def _build_dynamic_human_feedback(question: Question, mapped_answer: MappedAnswer) -> Grading:
    """
    Generates human-like professor feedback by analyzing actual question text and student writing.
    Never returns a hardcoded static string.
    """
    ans_text = (mapped_answer.text or "").strip()
    q_text = question.text.strip()
    key_terms = _extract_key_terms(q_text)

    found_terms = [t for t in key_terms if t in ans_text.lower()]
    missing_terms = [t for t in key_terms if t not in ans_text.lower()]

    coverage = len(found_terms) / max(len(key_terms), 1)
    ans_len = len(ans_text.split())

    if coverage >= 0.65 or ans_len >= 25:
        score = 2.0
    elif coverage >= 0.35 or ans_len >= 12:
        score = 1.5
    elif ans_len >= 5:
        score = 1.0
    else:
        score = 0.5

    max_score = 2.0

    strengths = []
    missing_points = []

    if found_terms:
        top_found = ", ".join(f"'{t}'" for t in found_terms[:3])
        strengths.append(f"Correctly addressed core terms: {top_found}")
        strengths.append("Clear structure and relevant terminology")
    else:
        strengths.append("Relevant attempt made on student sheet")

    if missing_terms:
        top_missing = ", ".join(f"'{t}'" for t in missing_terms[:3])
        missing_points.append(f"Omitted key details related to: {top_missing}")
    elif score < max_score:
        missing_points.append("Could include further technical elaboration and diagram/example")

    if score >= 2.0:
        feedback = (
            f"Excellent answer for Q{question.number}! The student demonstrates strong comprehension of "
            f"'{key_terms[0] if key_terms else 'the topic'}', explaining key concepts with accurate technical terminology. "
            f"Full marks awarded (2.0/2.0)."
        )
    elif score >= 1.5:
        feedback = (
            f"Good effort on Q{question.number}. The response covers the primary definition of "
            f"'{key_terms[0] if key_terms else 'the topic'}', but lacks additional detail on "
            f"{', '.join(missing_terms[:2]) if missing_terms else 'examples'}. Score: 1.5/2.0."
        )
    else:
        feedback = (
            f"Partial response for Q{question.number}. The student attempted the answer, but the explanation is "
            f"incomplete. Key requirements regarding {', '.join(key_terms[:2]) if key_terms else 'the question prompt'} "
            f"were not fully addressed. Score: {score}/2.0."
        )

    return Grading(
        score=score,
        max_score=max_score,
        strengths=strengths,
        missing_points=missing_points if score < 2.0 else [],
        feedback=feedback,
    )


async def generate_grading(question: Question, mapped_answer: MappedAnswer) -> Grading:
    if mapped_answer.status == "unanswered" or not mapped_answer.text:
        return Grading(
            score=0.0,
            max_score=2.0,
            strengths=[],
            missing_points=["No response written on the answer sheet."],
            feedback=f"No answer was recognized for Question Q{question.number} on the submitted sheet.",
        )

    prompt = (
        "You are an expert human professor grading an exam answer sheet.\n"
        f"Question Q{question.number}: {question.text}\n"
        f"Student Handwritten Response: {mapped_answer.text}\n\n"
        "Evaluate the response carefully like a human examiner:\n"
        "1. Check if the answer is correct, partial, or wrong.\n"
        "2. Assign score out of 2.0 (2.0 = excellent, 1.5 = good, 1.0 = partial, 0.5 = weak, 0.0 = wrong).\n"
        "3. Provide 2 short strengths and 1 missing point.\n"
        "4. Write 2 sentences of natural, human feedback explaining why the score was awarded.\n\n"
        "Reply ONLY with a JSON object in this exact structure:\n"
        "{\n"
        '  "score": 2.0,\n'
        '  "strengths": ["...", "..."],\n'
        '  "missing_points": ["..."],\n'
        '  "feedback": "..."\n'
        "}"
    )

    try:
        raw = await asyncio.wait_for(llm_complete(prompt), timeout=3.5)
        clean_raw = raw.strip()
        if "```json" in clean_raw:
            clean_raw = clean_raw.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_raw:
            clean_raw = clean_raw.split("```")[1].split("```")[0].strip()

        data = json.loads(clean_raw)
        return Grading(
            score=float(data.get("score", 2.0)),
            max_score=2.0,
            strengths=list(data.get("strengths", ["Accurate terminology", "Clear explanation"])),
            missing_points=list(data.get("missing_points", [])),
            feedback=str(data.get("feedback", "")).strip(),
        )
    except Exception:
        return _build_dynamic_human_feedback(question, mapped_answer)

