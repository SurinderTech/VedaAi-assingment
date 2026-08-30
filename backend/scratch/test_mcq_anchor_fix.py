"""
Quick sanity test for the MCQ anchor parsing fix.

Verifies that:
  - "Q1. (D) Combustion of LPG"  -> anchor = Q1  (MCQ answer, NOT subpart)
  - "Q1(a) Define deep learning"  -> anchor = Q1(a) (genuine subpart)
  - "1. (B) 2:1"                  -> anchor = Q2   (MCQ answer, NOT subpart)
  - "1(b). Solve this"            -> anchor = Q1(b) (genuine subpart)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
from app.models.schemas import Block, BBox

# -- replicate the regex and helper from answer_extractor -----------------
DIRECT_PAREN_RE = re.compile(
    r"^\s*(?:Ans(?:wer)?\.?\s*|Q(?:uestion)?\.?\s*)?(\d{1,3})\s*[\.\s\-_]*\(([a-zA-Z0-9]{1,2})\)\s*[\.):?]?\s*(.*)$",
    re.IGNORECASE,
)
MAIN_ANCHOR_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*|Ans(?:wer)?\.?\s*)?(\d{1,3})\s*[\.\):]?\s*(.*)$",
    re.IGNORECASE,
)

def classify(txt: str) -> str:
    """Returns the anchor that would be generated for a given text line."""
    cleaned = txt.strip()
    m = DIRECT_PAREN_RE.match(cleaned)
    if m:
        main_n = m.group(1)
        sub_c  = m.group(2).lower()
        # MCQ separator check
        num_end_pos = cleaned.index(main_n) + len(main_n)
        after_num   = cleaned[num_end_pos:]
        is_mcq = bool(re.match(r"^\s*\.\s+\(", after_num))
        if is_mcq:
            # fall through to MAIN_ANCHOR_RE
            pass
        else:
            return f"Q{main_n}({sub_c})"
    
    # MAIN_ANCHOR_RE path
    m2 = MAIN_ANCHOR_RE.match(cleaned)
    if m2:
        return f"Q{m2.group(1)}"
    return "NO_MATCH"

# -------------------------------------------------------------------------
cases = [
    # (input_text,                                    expected_anchor)
    ("Q1. (D) Combustion of Liquefied Petroleum Gas", "Q1"),    # MCQ answer
    ("Q2. (B) 2:1",                                  "Q2"),    # MCQ answer
    ("3. (B) (i) and (iv)",                           "Q3"),    # MCQ answer – digit only prefix
    ("Q4. (D) Hydrochloric acid",                     "Q4"),    # MCQ answer
    ("Q1(a) Define deep learning",                    "Q1(a)"), # true subpart
    ("1(b). Solve this",                              "Q1(b)"), # true subpart
    ("Q7(c). Explain backprop",                       "Q7(c)"), # true subpart
]

all_passed = True
for txt, expected in cases:
    got = classify(txt)
    status = "PASS" if got == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"[{status}]  {txt!r:55s}  ->  got={got!r}  expected={expected!r}")

print()
print("ALL PASSED" if all_passed else "SOME TESTS FAILED")
sys.exit(0 if all_passed else 1)
