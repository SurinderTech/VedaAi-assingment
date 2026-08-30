"""
Tests for the Group-Parent Guard in IntelligentQuestionExtractionService.

Verifies that intro-only/group-parent QUESTION nodes are rejected,
real questions are allowed, and section headers don't become questions.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- replicate the guard logic from the service --------------------
_group_parent_intro_re = re.compile(
    r'\b(?:write\s+briefly|answer\s+the\s+following|attempt\s+any|'
    r'short\s+answer\s+(?:questions?)?|attempt\s+the\s+following|'
    r'answer\s+any|describe\s+briefly|explain\s+briefly|'
    r'give\s+(?:short\s+)?(?:answers?|notes?))\b',
    re.IGNORECASE,
)

def should_reject_as_group_parent(q_text: str) -> bool:
    body = re.sub(r'^\s*(?:Q(?:uestion)?\.?\s*)?\d{1,3}\s*[.):–-]?\s*', '', q_text).strip()
    ends_with_colon = q_text.rstrip().endswith(':')
    is_intro_only = bool(_group_parent_intro_re.search(body) and len(body) < 80)
    return ends_with_colon or is_intro_only

# ---------- test cases ----------------------------------------------------
# (text, expected_reject)
cases = [
    # Should be REJECTED (group parent headers)
    ("1. Write briefly :",                                         True),
    ("Write briefly :",                                            True),
    ("1. Answer the following :",                                  True),
    ("2. Answer any FOUR of the following:",                       True),
    ("Q3. Give short notes on the following:",                     True),
    ("1. Short answer questions:",                                 True),
    ("1. Describe briefly:",                                       True),
    ("1. Explain briefly the following:",                          True),

    # Should be ALLOWED (real questions)
    ("1. What is photosynthesis?",                                 False),
    ("2. Explain the working of a transformer model.",             False),
    ("3. Define deep learning.",                                   False),
    ("4. Calculate the resistance of a wire of length 2m.",        False),
    ("5. Differentiate between bias and variance.",                False),
    ("6. What is an activation function? Explain about various components.",  False),
    ("7. Explain the back propagation algorithm with the help of an example.", False),
    # Section headers are handled elsewhere, but just in case:
    ("SECTION-A (COMPULSORY)",                                     False),  # no number prefix → body = whole text, no intro verb
]

all_passed = True
for txt, expected_reject in cases:
    got_reject = should_reject_as_group_parent(txt)
    status = "PASS" if got_reject == expected_reject else "FAIL"
    if status == "FAIL":
        all_passed = False
    tag = "REJECT" if expected_reject else "ALLOW "
    print(f"[{status}] [{tag}]  {txt!r}")

print()
print("ALL PASSED" if all_passed else "SOME TESTS FAILED")
sys.exit(0 if all_passed else 1)
