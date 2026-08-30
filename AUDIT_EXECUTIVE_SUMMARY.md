# EXECUTIVE SUMMARY: Q1 + 1(a)-1(j) Audit

## Status: ✓ CAPABLE BUT UNTESTED ON REAL Q1+SUBQUESTIONS

---

## Quick Answer: Can it work?

**YES** — all 5 layers fully support the required structure:

```
VLM Prompt:        ✓ Allows "SUBQUESTION" role and "subquestion_of" relationship
Schema:            ✓ Defines SUBQUESTION, subquestion_of, parent_question_id
Parser:            ✓ Accepts SUBQUESTION structures and subquestion_of edges
Graph:             ✓ Builds nodes with role=SUBQUESTION and edges with type=subquestion_of
Extraction:        ✓ Walks edges, creates Questions with parent_question_id linking
```

---

## One Critical Gap: Relationship Inference

**If VLM returns:**
- ✓ Structures with role="SUBQUESTION", display_number="a", "b", "c"
- ✓ Structures with role="QUESTION", display_number="1"
- ✓ Explicit relationship edges with type="subquestion_of"

**Then:** Everything works perfectly end-to-end.

**If VLM returns:**
- ✓ Structures with correct roles
- ✗ NO relationship edges (empty relationships array)

**Then:** Subquestions become orphaned nodes, hierarchy is lost.

**Root Cause:** Line 586 in `document_understanding_service.py` only applies explicit relationships. No fallback inference from display_number + role.

---

## Production Readiness: NOT YET

**Reason:** Only tested on synthetic 2-block test image, not real Q1+10-subquestion PDF.

**What's needed:**
1. Run real PDF with Q1 + 1(a)-1(j) through pipeline
2. Verify VLMPageUnderstanding.relationships contains subquestion_of edges
3. Verify DocumentStructureGraph.edges contains subquestion_of edges
4. Verify extracted_questions show parent_question_id hierarchy

**If all 3 pass:** Fix #2 is production-ready
**If any fails:** Specific layer is identified for fix

---

## MCQ vs Subquestion Distinction: FRAGILE

Deterministic regex (line 901-927 in extraction_service.py):
- ✓ Works if OCR preserves case: `(a)` vs `(A)`
- ✓ Works if layout is unambiguous
- ✗ Fails if OCR errors or ambiguous layout

**But:** VLM override compensates
- Can see visual structure, indentation, font size
- Should classify correctly based on image evidence

---

## Required Action: Test on Real Q1+Subquestions

**Location to test:** `backend/scratch/multi_page_paper.pdf`
- Current PDF has Q5, Q6, Q7 (NOT Q1 with subquestions)
- Need to locate or create PDF with Q1 + 1(a)-1(j)

**Diagnostic to run:**
```python
# Once real Q1+subquestions PDF found:
from app.services.document_processor import process_document
from app.services.document_understanding_service import DocumentUnderstandingService

blocks, pages, sizes, images = process_document(pdf_path, ".pdf")
service = DocumentUnderstandingService()
result = service.process_document(blocks, page_sizes=sizes, page_images=images, force_vlm_verification=True)

# Check 1: VLM relationships
print("VLM relationships:", result.vlm_page_understandings[0].relationships)
# Expected: Contains {source: sub_a_id, target: q1_id, type: "subquestion_of"}

# Check 2: Graph edges
print("Graph edges:", result.structure_graph.edges)
# Expected: Contains edges with relationship="subquestion_of"

# Check 3: Extracted hierarchy
extractor = IntelligentQuestionExtractService()
extraction = extractor.extract_validated_questions(blocks, doc_understanding_result=result)
for q in extraction.questions:
    if q.parent_question_id:
        print(f"Subquestion {q.number}: parent={q.parent_question_id}")
# Expected: 1(a), 1(b), ..., 1(j) all have parent_question_id pointing to Q1
```

---

## No Code Changes Recommended Yet

**Reason:** All required capabilities exist in code. Only question is whether VLM will provide explicit relationships.

**Wait for:** Real PDF test results before deciding if defensive inference is needed.

---

## Files Involved

| File | Component | Status |
|------|-----------|--------|
| schemas.py | SUBQUESTION role, subquestion_of relationship | ✓ Complete |
| document_vision_provider.py | VLM prompt, parser | ✓ Complete |
| document_understanding_service.py | Graph construction, VLM application | ✓ Complete (+ 1 gap) |
| intelligent_question_extraction_service.py | Graph-driven extraction, hierarchy | ✓ Complete |

---

## Summary

✓ **Architecture:** Fully supports Q1 + 1(a)-1(j)
✓ **Code quality:** All layers properly implemented
⚠️ **Robustness:** Missing defensive relationship inference if VLM fails
✗ **Verification:** Only synthetic test, not real Q1+subquestions PDF
✗ **Production:** NOT ready until real PDF tested

**Next step:** Find/create real Q1+10-subquestions PDF and run diagnostic.

