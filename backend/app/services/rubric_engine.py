"""
Rubric Engine.

Generates structured rubrics with criterion-level weights (max_marks) based on question requirement specs.
Uses deterministic generation for straightforward questions and structured LLM criteria proposals for complex questions.
Enforces strict normalization so sum(criterion.max_marks) == total_max_marks.
"""
from __future__ import annotations
import asyncio
from typing import List
from app.models.schemas import Question, QuestionRequirementSpec, Rubric, RubricCriterion
from app.services.llm_provider import llm_complete_json


def generate_deterministic_rubric(question: Question, spec: QuestionRequirementSpec) -> Rubric:
    """Generates a structured rubric deterministically without calling LLM."""
    total_marks = spec.max_marks or 2.0
    q_type = spec.expected_answer_type
    
    criteria: List[RubricCriterion] = []
    
    if q_type == "mcq":
        opt_desc = f"Selects correct option ({question.correct_option}): {question.correct_answer}" if question.correct_option else "Selects the correct option choice"
        criteria.append(RubricCriterion(id="c1", description=opt_desc, max_marks=total_marks))
    elif q_type == "one_word":
        ans_desc = f"Provides accurate factual name or value: {question.correct_answer}" if question.correct_answer else "Provides accurate factual name or value"
        criteria.append(RubricCriterion(id="c1", description=ans_desc, max_marks=total_marks))
    elif q_type == "definition":
        concepts_str = ", ".join(spec.required_concepts[:2]) if spec.required_concepts else "target concept"
        def_desc = f"Defines the core concept: {question.correct_answer}" if question.correct_answer else f"Defines the core concept of {concepts_str} correctly"
        criteria.append(RubricCriterion(id="c1", description=def_desc, max_marks=round(total_marks * 0.7, 2)))
        criteria.append(RubricCriterion(id="c2", description=f"Uses technical terminology related to {concepts_str}", max_marks=round(total_marks * 0.3, 2)))
    elif spec.has_numerical_requirement:
        num_desc = f"Correct final numerical value and units: {question.correct_answer}" if question.correct_answer else "Correct final numerical value and units"
        criteria.append(RubricCriterion(id="c1", description="Correct formula and step-by-step calculation method", max_marks=round(total_marks * 0.6, 2)))
        criteria.append(RubricCriterion(id="c2", description=num_desc, max_marks=round(total_marks * 0.4, 2)))
    elif spec.has_diagram_requirement:
        criteria.append(RubricCriterion(id="c1", description="Required diagram structure, components, and labels", max_marks=round(total_marks * 0.5, 2)))
        criteria.append(RubricCriterion(id="c2", description="Accurate text explanation of the diagram", max_marks=round(total_marks * 0.5, 2)))
    elif question.key_points and len(question.key_points) >= 2:
        # Use ground truth key points for conceptual questions
        num_kp = len(question.key_points)
        per_mark = round(total_marks / num_kp, 2)
        for i, kp in enumerate(question.key_points):
            criteria.append(RubricCriterion(id=f"c{i+1}", description=f"Addresses key point: {kp}", max_marks=per_mark))
    else:
        # Generic multi-concept split
        descs = spec.evaluation_criteria_descriptions or ["States core concept", "Explains supporting mechanism"]
        num_c = max(1, len(descs))
        per_mark = round(total_marks / num_c, 2)
        for i, d in enumerate(descs):
            criteria.append(RubricCriterion(id=f"c{i+1}", description=d, max_marks=per_mark))
            
    # Ensure sum of max_marks equals total_marks exactly
    c_sum = sum(c.max_marks for c in criteria)
    if abs(c_sum - total_marks) > 1e-3 and criteria:
        criteria[-1].max_marks = round(criteria[-1].max_marks + (total_marks - c_sum), 2)
        
    return Rubric(answer_type=q_type, total_max_marks=total_marks, criteria=criteria)


async def generate_rubric(question: Question, spec: QuestionRequirementSpec) -> Rubric:
    """
    Generates a structured rubric. For complex questions, requests LLM criteria proposals,
    then normalizes and validates them deterministically.
    """
    det_rubric = generate_deterministic_rubric(question, spec)
    total_marks = spec.max_marks or 2.0
    
    # Simple questions do NOT need LLM rubric generation (Token Saving)
    if spec.expected_answer_type in ("mcq", "one_word", "definition") or total_marks <= 2.0:
        return det_rubric
        
    prompt = (
        "You are an exam rubric generator.\n"
        f"Question: {question.text}\n"
        f"Question Type: {spec.expected_answer_type}\n"
        f"Total Max Marks: {total_marks}\n\n"
        "Propose 3 to 5 distinct grading criteria for this question.\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "criteria": [\n'
        '    {"id": "c1", "description": "Explains concept X", "max_marks": 2.5},\n'
        '    {"id": "c2", "description": "Provides step Y", "max_marks": 2.5}\n'
        "  ]\n"
        "}"
    )
    
    try:
        data = await asyncio.wait_for(llm_complete_json(prompt), timeout=4.0)
        if isinstance(data, dict) and "criteria" in data and isinstance(data["criteria"], list):
            llm_criteria = []
            for idx, raw_c in enumerate(data["criteria"]):
                if isinstance(raw_c, dict) and raw_c.get("description"):
                    llm_criteria.append(
                        RubricCriterion(
                            id=f"c{idx+1}",
                            description=str(raw_c["description"]).strip(),
                            max_marks=float(raw_c.get("max_marks", total_marks / len(data["criteria"])))
                        )
                    )
            if llm_criteria:
                # Normalize sum to match total_marks exactly
                tot = sum(c.max_marks for c in llm_criteria)
                if tot > 0:
                    for c in llm_criteria:
                        c.max_marks = round((c.max_marks / tot) * total_marks, 2)
                    c_sum = sum(c.max_marks for c in llm_criteria)
                    llm_criteria[-1].max_marks = round(llm_criteria[-1].max_marks + (total_marks - c_sum), 2)
                return Rubric(answer_type=spec.expected_answer_type, total_max_marks=total_marks, criteria=llm_criteria)
    except Exception:
        pass
        
    return det_rubric
