"""
Universal Exam Paper Answer Key & Ground Truth Resolution Engine.

Universal Capabilities:
1. Resolves authoritative ground truth for ANY question paper regardless of subject,
   class, board, format, or question count (CBSE, ICSE, University, Medical, Engineering, etc.).
2. Solves all question types:
   - MCQs: correct_option letter, full option text, explanation.
   - Numerical: exact numerical answer, units, formula, intermediate values.
   - Short Answer / One-Word: canonical factual answer, accepted synonyms/variations.
   - Long Conceptual / Definition / Derivation: concise model answer, key mandatory concepts.
3. Scalable Batch Chunking:
   Processes large question papers (e.g., 50-100 questions) in concurrent chunks of 15-20
   questions to ensure zero token truncation and sub-3-second response times.
"""
from __future__ import annotations

import re
import asyncio
from typing import List, Dict, Any, Optional

from app.models.schemas import Question
from app.services.llm_provider import llm_complete_json


CHUNK_SIZE = 18


def _clean_question_number(num: str) -> str:
    s = str(num).strip()
    if s.lower().startswith("q"):
        s = s[1:].strip().lstrip(".")
    return s.rstrip(".:-").strip()


async def _solve_question_chunk(chunk: List[Question]) -> Dict[str, Dict[str, Any]]:
    """Solves a single chunk of questions using LLM structured completion."""
    items = []
    for q in chunk:
        q_entry: Dict[str, Any] = {
            "number": str(q.number),
            "text": q.text,
            "question_type": q.question_type if q.question_type != "UNKNOWN" else "AUTO_DETECT",
            "max_marks": q.max_marks or 2.0,
        }
        if q.options and len(q.options) > 0:
            q_entry["options"] = q.options
        items.append(q_entry)

    prompt = (
        "You are an expert universal exam evaluator and master teacher.\n"
        "Solve the following examination questions with 100% academic precision.\n\n"
        "For EACH question:\n"
        "- If Multiple Choice (MCQ): Provide 'correct_option' (single uppercase letter like A, B, C, D) and 'correct_answer' (text of that option).\n"
        "- If Numerical: Provide 'correct_answer' containing the exact final numeric value and SI unit (e.g., '0.25 m', '15 ohms', '9.8 m/s²') and 'key_points' listing formula and intermediate steps.\n"
        "- If Short Answer / One Word / Factual: Provide 'correct_answer' (canonical name/value) and 'key_points' (accepted synonyms/terms).\n"
        "- If Long Conceptual / Definition: Provide 'correct_answer' (concise 2-sentence model answer) and 'key_points' (2 to 4 mandatory concepts).\n\n"
        f"Questions:\n{items}\n\n"
        "Return ONLY a JSON object with this exact structure (no markdown fences, no conversational text):\n"
        "{\n"
        '  "solutions": [\n'
        '    {\n'
        '      "number": "21",\n'
        '      "correct_option": "A",\n'
        '      "correct_answer": "A convex lens has 4 dioptre power having a focal length 0.25 m",\n'
        '      "explanation": "P = 1/f = 1/+0.25m = +4 D for a convex lens.",\n'
        '      "key_points": ["P = 1/f", "convex lens has positive focal length", "power is +4D"]\n'
        '    }\n'
        "  ]\n"
        "}"
    )

    try:
        data = await asyncio.wait_for(llm_complete_json(prompt), timeout=25.0)
        if isinstance(data, dict) and "solutions" in data and isinstance(data["solutions"], list):
            res_map = {}
            for item in data["solutions"]:
                if isinstance(item, dict) and item.get("number"):
                    n_key = _clean_question_number(str(item["number"]))
                    res_map[n_key] = item
            return res_map
    except Exception as e:
        print(f"[UniversalAnswerKey] Error solving chunk: {e}")

    return {}


async def resolve_universal_answer_key(questions: List[Question]) -> None:
    """
    Universal entry point: Solves and enriches all questions in a question paper with
    authoritative ground truth answers, options, and key points before grading.
    """
    if not questions:
        return

    # Identify questions that need solving (preserve any pre-existing solutions)
    to_solve = [q for q in questions if not q.correct_answer and not q.correct_option]
    if not to_solve:
        return

    print(f"[UniversalAnswerKey] Resolving ground truth solutions for {len(to_solve)}/{len(questions)} question(s)...")

    # Chunk into manageable batches to avoid token limits
    chunks = [to_solve[i:i + CHUNK_SIZE] for i in range(0, len(to_solve), CHUNK_SIZE)]
    tasks = [_solve_question_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    solved_count = 0
    for chunk_res in results:
        if isinstance(chunk_res, dict):
            for q in to_solve:
                q_num = _clean_question_number(q.number)
                if q_num in chunk_res:
                    sol = chunk_res[q_num]
                    c_opt = str(sol.get("correct_option", "")).strip().upper()
                    if c_opt and c_opt in ("A", "B", "C", "D", "E", "I", "II", "III", "IV"):
                        q.correct_option = c_opt
                    q.correct_answer = str(sol.get("correct_answer", "")).strip() or q.correct_answer
                    q.explanation = str(sol.get("explanation", "")).strip() or q.explanation
                    raw_kp = sol.get("key_points", [])
                    if isinstance(raw_kp, list):
                        q.key_points = [str(k).strip() for k in raw_kp if str(k).strip()]
                    solved_count += 1

    print(f"[UniversalAnswerKey] Successfully enriched {solved_count} question(s) with ground truth.")
