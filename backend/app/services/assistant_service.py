"""
Context-Aware VedaAI Teacher Assistant Service (plan sections 20-25).

Provides structured assessment context to the LLM provider router
and extracts actionable question references (unanswered, low-confidence).
"""
from __future__ import annotations
from typing import Optional, Tuple, List
from app.core import store
from app.models.schemas import AssistantResponse
from app.services.llm_provider import llm_complete, LLMError


async def process_assistant_message(
    assessment_id: Optional[str], message: str, question_id: Optional[str] = None
) -> AssistantResponse:
    result = store.get_result(assessment_id) if assessment_id else None

    unanswered_q: List[str] = []
    review_q: List[str] = []
    attention_q: List[str] = []

    if result:
        unanswered_q = [q.number for q in result.questions if q.answer.status == "unanswered"]
        review_q = [
            q.number for q in result.questions if q.answer.status in ("review_required", "unmatched")
        ]
        attention_q = list(set(unanswered_q + review_q))

    # Fast-path for structured preset queries if offline/no LLM
    msg_lower = message.lower().strip()

    if result is not None:
        if "unanswered" in msg_lower:
            reply = (
                f"There are **{len(unanswered_q)} unanswered questions** in this assessment: "
                + (", ".join(f"**Q{q}**" for q in unanswered_q) if unanswered_q else "None! All questions were answered.")
            )
            return AssistantResponse(
                reply=reply,
                unanswered_questions=unanswered_q,
                attention_questions=attention_q,
                review_questions=review_q,
            )

        if "attention" in msg_lower or "need review" in msg_lower or "mapping" in msg_lower:
            reply = (
                f"Here is what needs your attention:\n"
                f"• **Unanswered Questions ({len(unanswered_q)}):** {', '.join(unanswered_q) if unanswered_q else 'None'}\n"
                f"• **Low-Confidence / Review Mappings ({len(review_q)}):** {', '.join(review_q) if review_q else 'None'}\n"
                f"• **Unmatched Student Writings:** {len(result.unmatched_answers)} blocks"
            )
            return AssistantResponse(
                reply=reply,
                unanswered_questions=unanswered_q,
                attention_questions=attention_q,
                review_questions=review_q,
            )

    # Build full prompt with assessment context for LLM
    context_str = ""
    if result:
        q_summary = []
        for q in result.questions:
            ans = q.answer
            conf = f"{int(ans.confidence * 100)}%" if ans.confidence else "N/A"
            text_snippet = (ans.text[:60] + "...") if ans.text else "No response"
            q_summary.append(
                f"- Q{q.number}: {q.text[:80]} | Status: {ans.status} ({conf}) | Student written: '{text_snippet}'"
            )

        context_str = (
            f"Assessment ID: {result.assessment_id}\n"
            f"Total Questions Detected: {len(result.questions)}\n"
            f"Unanswered Questions: {', '.join(unanswered_q) if unanswered_q else 'None'}\n"
            f"Review Recommended Questions: {', '.join(review_q) if review_q else 'None'}\n"
            f"Unmatched Student Answer Blocks: {len(result.unmatched_answers)}\n"
            f"Questions Overview:\n" + "\n".join(q_summary) + "\n"
        )
        if question_id:
            sel = next((q for q in result.questions if q.id == question_id), None)
            if sel:
                context_str += (
                    f"\nCurrently Selected Question by Teacher: Q{sel.number}\n"
                    f"Question Text: {sel.text}\n"
                    f"Mapped Answer Status: {sel.answer.status}\n"
                    f"Extracted Answer Text: {sel.answer.text}\n"
                )
    else:
        context_str = "Context: Teacher is on the main VedaAI dashboard / landing workspace.\n"

    system_instruction = (
        "You are VedaAI Teacher Assistant, a helpful context-aware AI co-pilot for exam evaluation.\n"
        "Answer the teacher's question accurately using the provided structured assessment data.\n"
        "Be concise, clear, and professional. Mention specific question numbers (e.g. Q11(a)) when relevant.\n\n"
        f"--- STRUCTURED ASSESSMENT DATA ---\n{context_str}\n"
        f"--- TEACHER QUESTION ---\n{message}\n"
    )

    try:
        raw_reply = await llm_complete(system_instruction)
        return AssistantResponse(
            reply=raw_reply.strip(),
            attention_questions=attention_q,
            unanswered_questions=unanswered_q,
            review_questions=review_q,
        )
    except LLMError:
        # Fallback response using local context if LLM keys fail
        fallback = (
            f"Welcome to VedaAI Teacher Assistant!\n"
            f"Upload a Question Paper and Student Answer Sheet to begin context-aware evaluation."
        )
        if result:
            fallback = (
                f"Here is your assessment summary:\n"
                f"• **Total Questions:** {len(result.questions)}\n"
                f"• **Unanswered:** {', '.join(unanswered_q) if unanswered_q else 'None'}\n"
                f"• **Needs Review:** {', '.join(review_q) if review_q else 'None'}"
            )
        return AssistantResponse(
            reply=fallback,
            attention_questions=attention_q,
            unanswered_questions=unanswered_q,
            review_questions=review_q,
        )
