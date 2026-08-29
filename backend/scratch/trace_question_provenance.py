import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core import store

def trace_provenance(aid: str):
    res = store.get_result(aid)
    if not res:
        print(f"No result found for assessment {aid}")
        return

    print("================================================================================")
    print(f" QUESTION PROVENANCE & GRAPH TRACE REPORT — {aid}")
    print("================================================================================")
    
    # Check if doc understanding result exists in store or disk
    # Let's inspect stored questions
    print(f"Total Extracted Questions in AssessmentResult: {len(res.questions)}\n")

    print(f"{'Idx':<4} | {'Question ID':<22} | {'Num':<8} | {'Page':<5} | {'Source Region IDs'}")
    print("-" * 80)
    for idx, q in enumerate(res.questions, 1):
        print(f"{idx:02d}   | {q.id:<22} | {q.number:<8} | {q.page:<5} | {getattr(q, 'source_region_ids', 'N/A')}")

if __name__ == "__main__":
    aid = sys.argv[1] if len(sys.argv) > 1 else "fresh_1b46df34"
    trace_provenance(aid)
