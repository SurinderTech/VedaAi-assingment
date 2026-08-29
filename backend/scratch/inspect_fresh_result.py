import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core import store

def print_assessment_summary(aid: str):
    res = store.get_result(aid)
    status = store.get_status(aid)
    if not res:
        print(f"No result found for assessment {aid}")
        return

    print(f"================================================================================")
    print(f" ASSESSMENT RESULT REPORT — {aid}")
    print(f" State: {status.state if status else 'N/A'} | Progress: {status.progress if status else 'N/A'}")
    print(f"================================================================================")
    print(f"Total Extracted Questions: {len(res.questions)}")

    total_score = sum(q.grading.score for q in res.questions if q.grading and q.grading.score is not None)
    max_score = sum(q.grading.max_score for q in res.questions if q.grading and q.grading.max_score is not None)
    print(f"Total Assessment Score: {total_score} / {max_score}\n")

    print(f"{'Idx':<4} | {'Question ID':<26} | {'Num':<8} | {'Type':<12} | {'Ans Status':<15} | {'Score':<6} | {'Mapped Regions'}")
    print("-" * 95)

    for idx, q in enumerate(res.questions, 1):
        ans = q.answer
        mapped_regions = len(ans.regions) if ans else 0
        score_val = q.grading.score if q.grading else 0.0
        max_val = q.grading.max_score if q.grading else 2.0
        score_str = f"{score_val:.1f}/{max_val:.1f}"
        status_str = ans.status if ans else "unanswered"
        print(f"{idx:02d}   | {q.id:<26} | {q.number:<8} | {'Q':<12} | {status_str:<15} | {score_str:<6} | {mapped_regions}")

    print("\n--- DETAILED QUESTION & MAPPED ANSWER TRACE ---")
    for idx, q in enumerate(res.questions, 1):
        ans = q.answer
        print(f"\n[{idx:02d}] Question #{q.number} (ID: {q.id})")
        print(f"     Page: {q.page} | Section: {q.section or 'None'}")
        print(f"     Text: {q.text[:100]}")
        if q.options:
            print(f"     Options ({len(q.options)}): {q.options}")
        if ans:
            print(f"     Answer Status: {ans.status}")
            print(f"     AS Region Count: {len(ans.regions)}")
            print(f"     Student Answer: {(ans.text or '')[:100]}")

if __name__ == "__main__":
    aid = sys.argv[1] if len(sys.argv) > 1 else "fresh_b10573fb"
    print_assessment_summary(aid)
