import sys
import os
import uuid
import asyncio
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.pipeline import run_pipeline
from app.core.store import save_files, get_result, get_status

qp_path = r"C:\Users\surin\AppData\Local\Temp\vedaai_uploads\5e3f68fc1759\question_paper.pdf"
as_path = r"C:\Users\surin\AppData\Local\Temp\vedaai_uploads\5e3f68fc1759\answer_sheet.pdf"

async def main():
    assessment_id = f"fresh_{uuid.uuid4().hex[:8]}"
    print(f"Creating FRESH assessment run: {assessment_id}")
    print(f"  QP: {qp_path}")
    print(f"  AS: {as_path}\n")

    save_files(assessment_id, qp_path, as_path, ".pdf", ".pdf")

    print(f"Launching pipeline processing for assessment {assessment_id}...")
    await run_pipeline(assessment_id)

    status = get_status(assessment_id)
    print(f"\nFinal Assessment Status: state={status.state}, progress={status.progress}, message={status.message}")

    result = get_result(assessment_id)
    if not result:
        print("ERROR: No result returned for assessment!")
        return

    print("\n================================================================================")
    print(f" FRESH ASSESSMENT RESULT — {assessment_id}")
    print("================================================================================")
    print(f"Total Extracted Questions: {len(result.questions)}")
    total_score = sum(q.answer.score for q in result.questions if q.answer and q.answer.score)
    max_score = sum(q.max_score for q in result.questions if q.max_score)
    print(f"Overall Assessment Score: {total_score} / {max_score}")

    print("\n--- EXTRACTED QUESTIONS & MAPPED ANSWERS ---")
    for idx, q in enumerate(result.questions, 1):
        ans = q.answer
        mapped_regions = len(ans.source_region_ids) if ans else 0
        score_str = f"{ans.score}/{q.max_score}" if ans else "N/A"
        status_str = ans.status if ans else "unanswered"
        print(f"[{idx:02d}] ID: {q.id:<25} | Num: {q.number:<8} | Type: {q.question_type:<12} | Status: {status_str:<15} | Score: {score_str:<6} | Mapped Regions: {mapped_regions}")
        print(f"     Text: {q.text[:80]}")
        if ans and ans.text:
            print(f"     Student Answer Snippet: {ans.text[:80]}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
