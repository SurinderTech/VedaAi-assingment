"""
Batch Intelligence Evaluation Runner.

Executes the Multimodal Document Intelligence Core across 5 target test documents:
1. qp.png (actual failing question-paper image)
2. qp8.png (actual question-paper image)
3. multi_page_paper.pdf (multi-page PDF document)
4. digital_sectioned_mcq.png (MCQ-heavy sectioned paper)
5. admin_heavy.png (administrative-heavy paper)

Outputs detailed VLM structural analysis, page roles, structure graph nodes/edges,
option/section/continuation relationships, and final graph-driven question extraction.
"""
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scratch.run_comprehensive_diagnostic import run_diagnostic

TEST_DOCUMENTS = [
    os.path.join(os.path.dirname(__file__), "qp.png"),
    os.path.join(os.path.dirname(__file__), "qp8.png"),
    os.path.join(os.path.dirname(__file__), "test_corpus", "multi_page_paper.pdf"),
    os.path.join(os.path.dirname(__file__), "test_corpus", "digital_sectioned_mcq.png"),
    os.path.join(os.path.dirname(__file__), "test_corpus", "admin_heavy.png"),
]

def main():
    print("#" * 80, flush=True)
    print(" VEDAAI REAL MULTIMODAL DOCUMENT INTELLIGENCE — BATCH EVALUATION REPORT", flush=True)
    print("#" * 80, flush=True)

    for idx, doc_path in enumerate(TEST_DOCUMENTS, 1):
        print(f"\n\n{'='*80}", flush=True)
        print(f" EVALUATION {idx}/5: {os.path.basename(doc_path)}", flush=True)
        print(f"{'='*80}", flush=True)

        if not os.path.exists(doc_path):
            print(f"ERROR: Document missing at {doc_path}", flush=True)
            continue

        run_diagnostic(doc_path, force_vlm=True)

if __name__ == "__main__":
    main()
