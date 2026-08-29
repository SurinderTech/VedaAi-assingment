import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService

qp_path = r"C:\Users\surin\AppData\Local\Temp\vedaai_uploads\5e3f68fc1759\question_paper.pdf"

def trace():
    print("================================================================================")
    print(" STEP-BY-STEP QUESTION PAPER GRAPH & EXTRACTION TRACE")
    print("================================================================================")
    
    blocks, num_pages, sizes, _ = process_document(qp_path, ".pdf")
    print(f"Total Ingested OCR Blocks: {len(blocks)}")
    
    # 2. Document Understanding & Graph Building
    dus = DocumentUnderstandingService()
    dur = dus.process_document(blocks, document_id="qp_trace")
    graph = dur.structure_graph

    print(f"\nTotal Graph Nodes: {len(graph.nodes)}")
    print(f"Total Graph Edges: {len(graph.edges)}")

    roles_count = {}
    for node in graph.nodes.values():
        roles_count[node.role] = roles_count.get(node.role, 0) + 1
    print("\nGraph Node Roles Breakdown:")
    for role, count in roles_count.items():
        print(f"  - {role:<18}: {count}")

    print("\nQUESTION Nodes in DocumentStructureGraph:")
    q_nodes = [n for n in graph.nodes.values() if n.role == "QUESTION"]
    for qn in q_nodes:
        print(f"  Node ID: {qn.region_id:<12} | Page: {qn.page} | Text: '{qn.text[:70]}'")

    print("\nAll Edges in DocumentStructureGraph:")
    for edge in graph.edges:
        print(f"  {edge.source_id} --[{edge.relationship}]--> {edge.target_id} (conf={edge.confidence})")

    # 3. Intelligent Question Extraction
    iqes = IntelligentQuestionExtractionService(doc_understanding_service=dus)
    result = iqes.extract_validated_questions(blocks, document_id="qp_trace", doc_understanding_result=dur)

    print("\n================================================================================")
    print(f" FINAL EXTRACTED QUESTIONS IN DocumentQuestionExtractionResult: {len(result.questions)}")
    print("================================================================================")
    print(f"Fallback Used: {result.fallback_used}")
    for idx, q in enumerate(result.questions, 1):
        print(f"{idx:02d}. ID: {q.id:<25} | Num: {q.number:<8} | Page: {q.page} | Source Regions: {q.source_region_ids}")
        print(f"    Text: {q.text[:80]}")

if __name__ == "__main__":
    trace()
