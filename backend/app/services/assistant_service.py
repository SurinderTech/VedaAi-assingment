"""
Context-Aware VedaAI Teacher Assistant Service.

Routes all AI assistant calls through the DEDICATED assistant_llm_provider
which uses its own isolated OpenRouter key — completely separate from the
main grading/VLM pipeline key in .env.
"""
from __future__ import annotations
from typing import Optional, List
from app.core import store
from app.models.schemas import AssistantResponse
from app.services.assistant_llm_provider import (
    assistant_llm_complete,
    AssistantLLMError,
)


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
            q.number for q in result.questions
            if q.answer.status in ("review_required", "unmatched")
        ]
        attention_q = list(set(unanswered_q + review_q))

    # ── Fast-path for structured preset queries (no LLM needed) ────────────
    msg_lower = message.lower().strip()

    if result is not None:
        if "unanswered" in msg_lower:
            reply = (
                f"There are **{len(unanswered_q)} unanswered questions** in this assessment: "
                + (", ".join(f"**Q{q}**" for q in unanswered_q) if unanswered_q else "None — all questions were answered! ✅")
            )
            return AssistantResponse(
                reply=reply,
                unanswered_questions=unanswered_q,
                attention_questions=attention_q,
                review_questions=review_q,
            )

        if "attention" in msg_lower or "need review" in msg_lower or "low-confidence" in msg_lower:
            reply = (
                f"Here is what needs your attention:\n"
                f"• **Unanswered Questions ({len(unanswered_q)}):** "
                f"{', '.join(f'Q{q}' for q in unanswered_q) if unanswered_q else 'None'}\n"
                f"• **Low-Confidence / Review Mappings ({len(review_q)}):** "
                f"{', '.join(f'Q{q}' for q in review_q) if review_q else 'None'}\n"
                f"• **Unmatched Student Writings:** {len(result.unmatched_answers)} blocks"
            )
            return AssistantResponse(
                reply=reply,
                unanswered_questions=unanswered_q,
                attention_questions=attention_q,
                review_questions=review_q,
            )

    # ── Build assessment context for the LLM ───────────────────────────────
    context_str = ""
    if result:
        q_summary = []
        for q in result.questions:
            ans = q.answer
            conf = f"{int(ans.confidence * 100)}%" if ans.confidence else "N/A"
            text_snippet = (ans.text[:80] + "...") if ans.text and len(ans.text) > 80 else (ans.text or "No response")
            q_summary.append(
                f"- Q{q.number}: {q.text[:100]} | Status: {ans.status} ({conf}) | Student wrote: '{text_snippet}'"
            )

        context_str = (
            f"Assessment ID: {result.assessment_id}\n"
            f"Total Questions Detected: {len(result.questions)}\n"
            f"Total Marks: {result.total_marks}\n"
            f"Unanswered Questions: {', '.join(f'Q{q}' for q in unanswered_q) if unanswered_q else 'None'}\n"
            f"Questions Needing Review: {', '.join(f'Q{q}' for q in review_q) if review_q else 'None'}\n"
            f"Unmatched Student Answer Blocks: {len(result.unmatched_answers)}\n"
            f"\nQuestions Overview:\n" + "\n".join(q_summary) + "\n"
        )

        if question_id:
            sel = next((q for q in result.questions if q.id == question_id), None)
            if sel:
                context_str += (
                    f"\nCurrently Selected Question: Q{sel.number}\n"
                    f"Question Text: {sel.text}\n"
                    f"Mapped Answer Status: {sel.answer.status}\n"
                    f"Extracted Answer Text: {sel.answer.text or 'Not available'}\n"
                )
    else:
        context_str = (
            "Context: The teacher is on the VedaAI main workspace.\n"
            "No assessment has been loaded yet. The teacher can upload a question paper and "
            "student answer sheet to begin context-aware evaluation.\n"
        )

    # ── Call the dedicated assistant LLM (isolated key) ────────────────────
    try:
        raw_reply = await assistant_llm_complete(
            user_message=message,
            assessment_context=context_str,
        )
        return AssistantResponse(
            reply=raw_reply.strip(),
            attention_questions=attention_q,
            unanswered_questions=unanswered_q,
            review_questions=review_q,
        )
    except AssistantLLMError as e:
        print(f"[AssistantService] All assistant LLM models failed: {e}")
        # Graceful fallback: answer from local context only
        if result:
            fallback = (
                f"Here is your assessment summary:\n"
                f"• **Total Questions:** {len(result.questions)}\n"
                f"• **Unanswered:** {', '.join(f'Q{q}' for q in unanswered_q) if unanswered_q else 'None'}\n"
                f"• **Needs Review:** {', '.join(f'Q{q}' for q in review_q) if review_q else 'None'}\n"
                f"• **Unmatched Answer Blocks:** {len(result.unmatched_answers)}\n\n"
                f"*(Full AI response temporarily unavailable — showing structured summary.)*"
            )
        else:
            fallback = (
                "I'm VedaAI Assistant — here to help you evaluate assessments.\n\n"
                "Upload a Question Paper and Student Answer Sheet to begin. "
                "I'll help you understand extracted questions, answer mappings, grading, and insights."
            )
        return AssistantResponse(
            reply=fallback,
            attention_questions=attention_q,
            unanswered_questions=unanswered_q,
            review_questions=review_q,
        )
