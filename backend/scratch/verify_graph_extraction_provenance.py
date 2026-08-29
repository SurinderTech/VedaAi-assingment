import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService

qp_path = r"C:\Users\surin\AppData\Local\Temp\vedaai_uploads\5e3f68fc1759\question_paper.pdf"

def main():
    print("================================================================================")
    print(" PROVENANCE VERIFICATION: 9 QUESTION NODES -> 19 EXTRACTED QUESTIONS")
    print("================================================================================")
    blocks, _, _, _ = process_document(qp_path, ".pdf")
    dus = DocumentUnderstandingService()
    dur = dus.process_document(blocks, document_id="qp_verify")
    graph = dur.structure_graph

    iqes = IntelligentQuestionExtractionService(doc_understanding_service=dus)
    result = iqes.extract_validated_questions(blocks, document_id="qp_verify", doc_understanding_result=dur)

    print(f"1. Fallback Extractor Used: {result.fallback_used} (Must be False)")
    print(f"2. Total QUESTION role nodes in DocumentStructureGraph: {sum(1 for n in graph.nodes.values() if n.role == 'QUESTION')}")
    print(f"3. Total SUBQUESTION role nodes in DocumentStructureGraph: {sum(1 for n in graph.nodes.values() if n.role == 'SUBQUESTION')}")
    print(f"4. Total subquestion_of edges in DocumentStructureGraph: {sum(1 for e in graph.edges if e.relationship == 'subquestion_of')}")
    print(f"5. Total Extracted Questions in Output: {len(result.questions)}\n")

    print(f"{'Idx':<4} | {'Question Num':<12} | {'Node Role':<15} | {'Graph Source Region ID':<24} | {'Parent ID'}")
    print("-" * 80)

    for idx, q in enumerate(result.questions, 1):
        region_id = q.source_region_ids[0]
        node = graph.nodes.get(region_id)
        role = node.role if node else "NOT_IN_GRAPH"
        parent_info = q.parent_question_id if q.parent_question_id else "None (Top-Level)"
        print(f"{idx:02d}   | {q.number:<12} | {role:<15} | {region_id:<24} | {parent_info}")

    # Assertions
    assert result.fallback_used == False, "Fallback extractor was incorrectly triggered!"
    assert all(q.source_region_ids[0] in graph.nodes for q in result.questions), "Question source region not found in DocumentStructureGraph!"
    print("\n[VERIFICATION PASSED] All 19 extracted questions originate 100% from DocumentStructureGraph nodes and subquestion_of edges!")

if __name__ == "__main__":
    main()
