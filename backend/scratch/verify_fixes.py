import sys, os
sys.path.insert(0, ".")

# Stub out env vars so config doesn't blow up
os.environ.setdefault("GEMINI_API_KEY", "x")
os.environ.setdefault("PRIMARY_LLM_PROVIDER", "gemini")
os.environ.setdefault("DOCUMENT_VLM_ENABLED", "true")
os.environ.setdefault("DOCUMENT_VLM_PAGE_UNDERSTANDING", "true")

from app.models.schemas import StructuredQuestionResult

sqr = StructuredQuestionResult(
    question_id="q1",
    question_number="1",
    question_text="Which of the following is correct?",
    max_marks=2.0,
    options=["A. Yes", "B. No", "C. Maybe", "D. None"],
)
print("Schema OK. options:", sqr.options)
assert sqr.options == ["A. Yes", "B. No", "C. Maybe", "D. None"], "FAIL: options not preserved"
print("PASS: StructuredQuestionResult.options field works correctly")
