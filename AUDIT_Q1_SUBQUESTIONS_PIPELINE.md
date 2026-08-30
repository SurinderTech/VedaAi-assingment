# COMPREHENSIVE AUDIT: Q1 + 1(a)-1(j) PIPELINE CAPABILITY

## STATUS: ARCHITECTURE COMPLETE, ONE DEFENSIVE GAP IDENTIFIED

---

## PART 1: VLM PROMPT AUDIT

**File:** `backend/app/services/document_vision_provider.py`
**Lines:** 195-268

### What the Prompt Tells VLM

```python
# Line 234-235
"role": "QUESTION" | "OPTION" | "SUBQUESTION" | "SECTION_HEADER" | ... | "UNKNOWN"

# Line 236-237
"type": "option_of" | "subquestion_of" | "section_member" | "continuation_of" | ...
```

### Explicit Capabilities ✓

The prompt:
- ✓ Lists "SUBQUESTION" as valid role
- ✓ Lists "subquestion_of" as valid relationship type
- ✓ Asks VLM to treat page image as authoritative for visual structure
- ✓ Provides complete OCR evidence with IDs and bboxes
- ✓ Says "Do NOT invent textual content. Preserve exact OCR text"

### Semantic Gaps ✗

The prompt:
- ✗ Does NOT explain how to distinguish (a), (b), (c) from A), B), C), D)
- ✗ Does NOT provide examples of hierarchical subquestion structures
- ✗ Does NOT explicitly teach "subquestion_of" parent-child semantics
- ✗ Relies on VLM's pre-trained knowledge to recognize subquestions

### Verdict

**Capability: ✓ SUFFICIENT at schema level**
**Robustness: ⚠️ DEPENDS on VLM training**

The prompt allows VLM to return subquestions with parent relationships, but does not guarantee it will correctly distinguish between subquestions and MCQ options.

---

## PART 2: SCHEMA AUDIT

**File:** `backend/app/models/schemas.py`

### VLMStructureItem (Lines 874-895)

```python
class VLMStructureItem(BaseModel):
    region_ids: List[str] = []                    # OCR blocks VLM directly referenced
    grounded_region_ids: List[str] = []           # OCR blocks matched by geometry
    grounding_status: str = "UNGROUNDED"          # Status after grounding
    grounded_text: str = ""                       # Assembled text from grounded regions
    bbox: Optional[BBox] = None                   # Visual bbox for visual-only structures
    role: DocumentRegionType = "UNKNOWN"          # QUESTION, SUBQUESTION, OPTION, etc.
    display_number: Optional[str] = None          # "1", "a", "b", "c", etc.
    display_label: Optional[str] = None           # "Question 1", "Part (a)", etc.
    confidence: float = 0.5
```

✓ **Supports:** SUBQUESTION role ✓ **Supports:** Optional parent info (display_number)

### VLMRelationshipItem (Lines 896-901)

```python
class VLMRelationshipItem(BaseModel):
    source_ids: List[str] = []        # Subquestion node IDs
    target_ids: List[str] = []        # Question node ID
    relationship_type: str = "belongs_to"  # Can be "subquestion_of"
    confidence: float = 0.5
```

✓ **Supports:** `subquestion_of` relationship type

### DocumentRegionType (Line 610)

```python
DocumentRegionType = Literal[
    "QUESTION",
    "SUBQUESTION",
    "OPTION",
    ...
]
```

✓ **Supports:** SUBQUESTION as region type

### RelationshipType (Line 628)

```python
RelationshipType = Literal[
    "subquestion_of",
    "option_of",
    ...
]
```

✓ **Supports:** `subquestion_of` as relationship type

### Question Model (Line 65)

```python
class Question(BaseModel):
    parent_question_id: Optional[str] = None      # Points to parent Q1
    question_type: QuestionType = "UNKNOWN"       # Includes "SUBQUESTION"
```

✓ **Supports:** Hierarchy via parent_question_id

### Verdict

**Capability: ✓ COMPLETE**

Schema fully supports representing Q1 + 1(a)-1(j) with proper parent-child relationships.

---

## PART 3: PARSER AUDIT

**File:** `backend/app/services/document_vision_provider.py`
**Method:** `_parse_page_understanding()` (Lines 474-553)

### Parse Flow

```
VLM JSON Response
    ↓
extract_json_payload()
    ↓
For each structure in JSON:
    - Get region_ids from VLM
    - Validate region_ids against real OCR blocks
    - Normalize role (e.g., "ADMINISTRATIVE" → "INSTRUCTION")
    - Check role is in valid_roles set
    ↓
Create VLMStructureItem IF (validated_ids exist OR bbox exists)
    ↓
For each relationship in JSON:
    - Validate source_ids and target_ids exist in OCR blocks
    - Check relationship_type is valid
    ↓
Create VLMRelationshipItem if both source and target valid
    ↓
Return VLMPageUnderstanding with structures + relationships
```

### Test Case: Subquestion Structure

**VLM returns:**
```json
{
  "role": "SUBQUESTION",
  "text": "(a) Define deep learning",
  "display_number": "a",
  "region_ids": ["ocr_block_123"],
  "bbox": [100, 500, 1500, 150],
  "confidence": 0.85
}
```

**Parser processes:**
1. Line 492: `raw_ids = item.get("region_ids", [])`  → `["ocr_block_123"]`
2. Line 493: `validated_ids = [rid for rid in raw_ids if rid in valid_ids]`
   - If "ocr_block_123" exists in OCR blocks → validated_ids = ["ocr_block_123"]
   - If not → validated_ids = []
3. Line 510: `if not validated_ids and parsed_bbox is None: continue`
   - If validated_ids is non-empty → **CONTINUE (process)**
   - If bbox exists → **CONTINUE (process)**
   - If both absent → **SKIP (filtered out)**
4. Line 505-508: Validate role
   - `normalized_role = "SUBQUESTION"` (already in valid_roles)
5. Line 514-524: Create VLMStructureItem
   ```python
   VLMStructureItem(
       region_ids=["ocr_block_123"],
       role="SUBQUESTION",
       display_number="a",
       confidence=0.85
   )
   ```

**Result: ✓ PARSED AND STORED**

### Test Case: Bbox-Only Subquestion (No OCR Region)

**VLM returns:**
```json
{
  "role": "SUBQUESTION",
  "display_number": "b",
  "bbox": [100, 700, 1500, 150],
  "confidence": 0.80
}
```

**Parser processes:**
1. Line 492: `raw_ids = []`
2. Line 493: `validated_ids = []` (no OCR blocks to validate)
3. Line 497-508: Parse bbox → `BBox(x=100, y=700, width=1500, height=150)`
4. Line 510: `if not validated_ids and parsed_bbox is None: continue`
   - validated_ids is empty BUT bbox is not None → **CONTINUE**
5. Line 514-524: Create VLMStructureItem
   ```python
   VLMStructureItem(
       region_ids=[],
       bbox=BBox(x=100, y=700, width=1500, height=150),
       role="SUBQUESTION",
       confidence=0.80
   )
   ```

**Result: ✓ PARSED AND STORED**

### Test Case: Subquestion with Bad Region IDs

**VLM returns:**
```json
{
  "role": "SUBQUESTION",
  "region_ids": ["invalid_id_xyz"],
  "confidence": 0.85
}
```

**Parser processes:**
1. Line 492: `raw_ids = ["invalid_id_xyz"]`
2. Line 493: `validated_ids = []` (invalid_id_xyz not in real OCR blocks)
3. Line 497: `parsed_bbox = None` (no bbox in VLM response)
4. Line 510: `if not validated_ids and parsed_bbox is None: continue`
   - Both empty → **SKIP (filtered out)**

**Result: ✗ REJECTED (but this is correct — prevents hallucinations)**

### Relationship Parsing

**VLM returns:**
```json
{
  "relationships": [
    {
      "source_ids": ["sub_a_ocr_block"],
      "target_ids": ["q1_ocr_block"],
      "type": "subquestion_of",
      "confidence": 0.9
    }
  ]
}
```

**Parser processes (line 526):**
```python
for rel in data.get("relationships", []):
    src_ids = [rid for rid in rel.get("source_ids", []) if rid in valid_ids]
    tgt_ids = [rid for rid in rel.get("target_ids", []) if rid in valid_ids]
    if not src_ids or not tgt_ids:
        continue  # Skip if either endpoint invalid
    
    rel_type = rel.get("type", "belongs_to")
    if rel_type not in valid_rel_types:
        rel_type = "belongs_to"
    
    relationships.append(VLMRelationshipItem(
        source_ids=src_ids,
        target_ids=tgt_ids,
        relationship_type=rel_type,
        confidence=0.9
    ))
```

**Result: ✓ PARSED AND STORED**

### Verdict

**Capability: ✓ COMPLETE**

Parser:
- ✓ Accepts SUBQUESTION role
- ✓ Accepts subquestion_of relationship
- ✓ Validates IDs against real OCR blocks
- ✓ Accepts bbox-only structures (Fix #2 capability)
- ✓ Creates VLMStructureItem and VLMRelationshipItem correctly

**Defensive behavior:**
- ✗ Rejects structures with invalid region_ids AND no bbox (correct — prevents hallucinations)

---

## PART 4: GRAPH CONSTRUCTION AUDIT

**File:** `backend/app/services/document_understanding_service.py`
**Method:** `_build_structure_graph()` (Lines 620-678)

### VLM Application Process

**Method:** `_apply_vlm_page_understandings()` (Lines 330-595)

#### Path A: VLM returns region_ids (Line 351)

```python
if struct.region_ids:
    head_id = struct.region_ids[0]
    cont_ids = struct.region_ids[1:]
    grounded_ids = list(struct.region_ids)
    grounding_status = "GROUNDED"
    
    # Update existing DocumentRegion
    reg.region_type = struct.role  # ← Set to SUBQUESTION
    reg.confidence = struct.confidence
    reg.verification_state = "VERIFIED"
```

**Result:** Existing OCR region's type is updated to SUBQUESTION

#### Path B: VLM returns bbox only (Line 356)

```python
else:
    grounded_ids, grounding_status, grounded_text = self._ground_structure_to_ocr(
        structure=struct,
        page_regions=[r for r in all_regions if r.page == understanding.page_number],
        page_number=understanding.page_number,
    )
    
    if grounded_ids:
        synthetic_region = DocumentRegion(
            region_id=head_id,
            region_type=struct.role,  # ← SUBQUESTION
            text=grounded_text,
            source="vlm_visual",
            ...
        )
        all_regions.append(synthetic_region)
```

**Result:** New synthetic region created with SUBQUESTION type

#### Relationships Application (Line 586)

```python
for rel in understanding.relationships:
    for src_id in rel.source_ids:
        for tgt_id in rel.target_ids:
            if src_id in region_map and tgt_id in region_map:
                all_relationships.append(RegionRelationship(
                    source_region_id=src_id,
                    target_region_id=tgt_id,
                    relationship_type=rel.relationship_type,  # "subquestion_of"
                    confidence=rel.confidence,
                    ...
                ))
```

**Result:** `RegionRelationship` created with type="subquestion_of"

### Graph Node Creation (Line 622)

```python
for r in all_regions:
    nodes[r.region_id] = GraphNode(
        region_id=r.region_id,
        role=r.region_type,  # SUBQUESTION
        text=r.text,
        page=r.page,
        bbox=r.bbox,
        confidence=r.confidence,
    )
```

### Graph Edge Creation (Line 630)

```python
for rel in all_relationships:
    edges.append(GraphEdge(
        source_id=rel.source_region_id,        # "sub_a_region"
        target_id=rel.target_region_id,        # "q1_region"
        relationship=rel.relationship_type,   # "subquestion_of"
        confidence=rel.confidence,
        evidence_sources=evidence_sources,
    ))
```

### Expected Graph Structure for Q1 + 1(a)-1(d)

```
Nodes:
  q1_ocr_123:
    role: QUESTION
    text: "Q1. Explain the concept of..."
    page: 1
    bbox: [100, 100, 1700, 200]
    confidence: 0.95

  sub_a_ocr_124:
    role: SUBQUESTION
    text: "(a) Define deep learning"
    page: 1
    bbox: [150, 320, 1600, 80]
    confidence: 0.90

  sub_b_ocr_125:
    role: SUBQUESTION
    text: "(b) Name three key algorithms"
    page: 1
    bbox: [150, 420, 1600, 80]
    confidence: 0.88

  ... (sub_c through sub_j)

Edges:
  sub_a_ocr_124 --[subquestion_of]--> q1_ocr_123
  sub_b_ocr_125 --[subquestion_of]--> q1_ocr_123
  sub_c_ocr_126 --[subquestion_of]--> q1_ocr_123
  ... (etc)
```

### Verdict

**Capability: ✓ COMPLETE**

Graph construction:
- ✓ Converts SUBQUESTION role to GraphNode with role="SUBQUESTION"
- ✓ Converts subquestion_of relationship to GraphEdge with relationship="subquestion_of"
- ✓ Preserves all metadata (bbox, page, text, confidence)
- ✓ Handles both OCR-grounded and bbox-only structures

---

## PART 5: EXTRACTION AUDIT

**File:** `backend/app/services/intelligent_question_extraction_service.py`
**Method:** `_extract_from_graph()` (Lines 155-388)

### Subquestion Extraction Process

#### Step 1: Build Edge Index (Line 186)

```python
children_of: Dict[str, List[Tuple[str, str, float]]] = {}
for edge in graph.edges:
    children_of.setdefault(edge.target_id, []).append(
        (edge.source_id, edge.relationship, edge.confidence)
    )
```

**Result for Q1:**
```python
children_of["q1_ocr_123"] = [
    ("sub_a_ocr_124", "subquestion_of", 0.90),
    ("sub_b_ocr_125", "subquestion_of", 0.88),
    ("sub_c_ocr_126", "subquestion_of", 0.85),
    ...
]
```

#### Step 2: Extract Question (Line 236)

```python
question_nodes = sorted(
    [n for n in graph.nodes.values() if n.role == "QUESTION"],
    key=lambda n: (n.page, n.bbox.y, n.bbox.x),
)

for q_node in question_nodes:
    # Create Question object for Q1
    q_obj = Question(
        id="doc:q1_ocr_123",
        number="1",
        text="Q1. Explain the concept of...",
        page=1,
        bbox=q_node.bbox,
        order_index=0,
        ...
    )
```

**Result:** Q1 stored in extracted_questions[0]

#### Step 3: Attach Subquestions (Line 303)

```python
self._attach_subquestions(
    parent_question=q_obj,
    question_node_id="q1_ocr_123",  # ← Q1 node ID
    children_of=children_of,        # ← Index of edges
    graph_nodes=graph.nodes,
    region_map=region_map,
    document_id="doc",
    extracted_questions=extracted_questions,
    order_counter_ref=[1],
    sec_title=None,
)
```

#### Step 4: Walk Subquestion Edges (Line 482)

**Method:** `_attach_subquestions()` (Lines 482-530)

```python
for child_id, rel_type, conf in children_of.get(question_node_id, []):
    if rel_type != "subquestion_of":  # ← Check relationship type
        continue
    
    child_node = graph_nodes.get(child_id)
    if not child_node or child_node.role != "SUBQUESTION":  # ← Check role
        continue
    
    # Extract display number from text: "(a) Define..." → "a"
    m = re.match(r"^\s*[\(\[]?\s*([a-z]{1,2})\s*[\)\]\.\:]\s*(.*)", 
                 child_node.text, re.IGNORECASE)
    sub_label = m.group(1).lower() if m else "?"
    parent_num = parent_question.number  # "1"
    display_num = f"{parent_num}({sub_label})"  # "1(a)"
    
    # Create subquestion
    sub_id = f"doc:{child_id}"
    sub_q = Question(
        id=sub_id,
        number=display_num,           # "1(a)", "1(b)", etc.
        text=child_node.text,         # "(a) Define deep learning"
        page=child_node.page,
        bbox=child_node.bbox,
        order_index=order_counter_ref[0],
        parent_question_id=parent_question.id,  # ← Links back to Q1
        question_type="SUBQUESTION",
        source_region_ids=[child_id],
        ...
    )
    
    extracted_questions.append(sub_q)
    order_counter_ref[0] += 1
```

### Expected Extraction Result for Q1 + 1(a)-1(j)

```python
extracted_questions = [
    Question(
        id="doc:q1_ocr_123",
        number="1",
        text="Q1. Explain...",
        question_type="SHORT_ANSWER",  # or LONG_ANSWER
        parent_question_id=None,
        order_index=0,
    ),
    Question(
        id="doc:sub_a_ocr_124",
        number="1(a)",
        text="(a) Define deep learning",
        question_type="SUBQUESTION",
        parent_question_id="doc:q1_ocr_123",  # ← Points to Q1
        order_index=1,
    ),
    Question(
        id="doc:sub_b_ocr_125",
        number="1(b)",
        text="(b) Name three algorithms",
        question_type="SUBQUESTION",
        parent_question_id="doc:q1_ocr_123",  # ← Points to Q1
        order_index=2,
    ),
    ...
    Question(
        id="doc:sub_j_ocr_133",
        number="1(j)",
        text="(j) Justify your answer",
        question_type="SUBQUESTION",
        parent_question_id="doc:q1_ocr_123",  # ← Points to Q1
        order_index=10,
    ),
]
```

### Invariant Validation (Line 355-368)

```python
if q.parent_question_id:
    parent_region_id = question_id_to_region_id.get(q.parent_question_id)
    if not parent_region_id or parent_region_id not in graph.nodes:
        audit.invariant_violations.append(
            f"Subquestion parent_question_id does not resolve to graph node"
        )
    else:
        parent_node = graph.nodes[parent_region_id]
        if parent_node.role != "QUESTION":
            audit.invariant_violations.append(
                f"Subquestion parent is not QUESTION role, got {parent_node.role}"
            )
        
        sub_region_id = q.source_region_ids[0]
        if edge_map.get((sub_region_id, parent_region_id)) != "subquestion_of":
            audit.invariant_violations.append(
                f"Subquestion missing 'subquestion_of' edge to parent"
            )
```

**Checks:**
1. ✓ parent_question_id resolves to actual node
2. ✓ Parent node has role="QUESTION"
3. ✓ Graph contains edge: (subquestion_node) → (question_node) with type="subquestion_of"

### Verdict

**Capability: ✓ COMPLETE**

Extraction:
- ✓ Walks graph edges for subquestion_of relationships
- ✓ Creates Question objects with parent_question_id
- ✓ Extracts display numbers correctly
- ✓ Validates invariants
- ✓ Handles 1(a), 1(b), ... 1(j) hierarchy

---

## PART 6: MCQ OPTION vs SUBQUESTION DISTINCTION

**File:** `backend/app/services/intelligent_question_extraction_service.py`
**Method:** `_analyze_region_hypotheses()` (Lines 887-970)

### Regex Patterns

```python
SUBQUESTION_PREFIX_RE = re.compile(r"^\(?([a-z]|[ivxlcdm]+)\)[\.\:\s]+", re.IGNORECASE)
OPTION_PREFIX_RE = re.compile(r"^\(?[A-D]\)[\.\:\s]+", re.IGNORECASE)

# In _analyze_region_hypotheses:
subq_match = re.search(r"^\(?([a-z]|[ivxlcdm]+)\)[\.\:\s]+", text, re.IGNORECASE)
opt_match = re.search(r"^\(?[A-Da-d1-9i-zIVXLCDM]+\)[\.\:\s]+", text, re.IGNORECASE)
```

### Deterministic Distinction Logic (Line 918-927)

```python
if subq_match and not q_num_match:
    parser_type = "SUBQUESTION"
    parser_conf = 0.82
elif opt_match and not interrogative_match:
    parser_type = "OPTION"
    parser_conf = 0.88
```

### How It Works

| Input | Pattern Match | Interrogative | Result | Confidence |
|-------|---------------|---------------|--------|------------|
| "(a) Define deep learning" | subq_match=Yes | No | SUBQUESTION | 0.82 |
| "(A) Option A" | opt_match=Yes | No | OPTION | 0.88 |
| "(i) Explain the theory" | subq_match=Yes (i=roman) | No | SUBQUESTION | 0.82 |
| "(I) Multiple choice A" | opt_match=Yes (I=option) | No | OPTION | 0.88 |
| "1. First question" | q_num_match=Yes | Yes | QUESTION | 0.92 |

### Weakness: Case Sensitivity

**Problem:** Regex distinguishes by case: `[a-z]` vs `[A-D]`

**OCR Errors:**
- PDF renders "(a)" but OCR outputs "(A)" → misclassified as OPTION
- PDF renders "(A)" but OCR outputs "(a)" → misclassified as SUBQUESTION

**Example Failure:**
```
PDF: Q1 with subquestions 1(a), 1(b), 1(c)
OCR (bad): "1(A) Define...", "1(B) Explain...", "1(C) Analyze..."
Deterministic: Classifies as OPTION (confidence 0.88)
VLM: Can see image and classify correctly as SUBQUESTION (confidence 0.9)
Result: VLM overrides deterministic classification ✓
```

### Semantic Distinction Capability

The deterministic regex **cannot** distinguish by context:
- ✗ Multi-line text under label (suggests subquestion)
- ✗ Single-line text under label (suggests option)
- ✗ Indentation level
- ✗ Font size relative to question

### VLM-Based Distinction

VLM can see:
- ✓ Page image with visual structure
- ✓ Indentation and spacing
- ✓ Font characteristics
- ✓ Multi-line nature of text
- ✓ Proximity to question number
- ✓ Page context and structure

### Verdict

**Deterministic Distinction: ⚠️ FRAGILE**
- Works if OCR perfectly preserves case
- Works if structure is unambiguous
- Fails under OCR errors or ambiguous layouts

**VLM-Based Distinction: ✓ ROBUST**
- Can distinguish based on visual evidence
- Can override deterministic errors
- Requires VLM to return correct role

**Combined (Deterministic + VLM): ✓ COMPLETE**
- Deterministic provides baseline hypothesis
- VLM overrides with higher confidence
- Graph uses VLM's role when available

---

## PART 7: CRITICAL GAP ANALYSIS

### Gap Identification

**Question:** Can the system extract Q1 + 1(a)-1(j) if VLM returns only roles but no relationships?

**Scenario:**

VLM returns:
```json
{
  "structures": [
    {"role": "QUESTION", "display_number": "1", "region_ids": ["q1_ocr_123"], ...},
    {"role": "SUBQUESTION", "display_number": "a", "region_ids": ["sub_a_ocr_124"], ...},
    {"role": "SUBQUESTION", "display_number": "b", "region_ids": ["sub_b_ocr_125"], ...},
    ... (all 10 subquestions)
  ],
  "relationships": []  # EMPTY — VLM didn't return relationships
}
```

**What happens:**

1. Parser creates VLMStructureItem for each structure ✓
2. Graph gets SUBQUESTION nodes with role="SUBQUESTION" ✓
3. **Graph gets NO edges between subquestions and question** ✗
4. Extraction sees subquestions as isolated nodes
5. _attach_subquestions() looks for edges with `rel_type == "subquestion_of"`
6. children_of["q1_ocr_123"] is empty
7. Loop body never executes
8. Subquestions are never attached to Q1
9. Subquestions extracted as independent questions with parent_question_id=None

**Result:** Q1+1(a)-1(j) hierarchy is **LOST**

### Root Cause

**File:** `document_understanding_service.py`
**Method:** `_apply_vlm_page_understandings()` (Line 586)

```python
for rel in understanding.relationships:  # ← Only applies explicit relationships
    for src_id in rel.source_ids:
        for tgt_id in rel.target_ids:
            all_relationships.append(RegionRelationship(...))
```

**Missing:** No fallback inference of parent-child relationships from:
- display_number ("1" vs "a", "b", "c")
- role (QUESTION vs SUBQUESTION)
- spatial proximity
- layout patterns

### Defensive Fix Location

**File:** `document_understanding_service.py`
**After:** Line 595 (end of _apply_vlm_page_understandings)

**What's needed:** Post-processing that infers `subquestion_of` edges when:
1. A node has role="SUBQUESTION" with display_number="a", "b", "c"
2. Another node has role="QUESTION" with display_number="1"
3. No explicit `subquestion_of` edge exists between them
4. They're on the same page
5. Subquestion is spatially below question

### Verdict

**Gap Type:** DEFENSIVE ROBUSTNESS, not capability gap

**Severity:** MEDIUM
- Schema supports it ✓
- Parser handles it ✓
- Graph handles it ✓
- Extraction handles it ✓
- **But:** Needs VLM to return explicit relationships or needs fallback inference

**Current Risk:** If VLM returns structures but no relationships, hierarchy is lost

**Mitigation Available:**
- Option A: VLM must return relationships (should already do this)
- Option B: Add inference logic (safer, defensive)
- Option C: Both (most robust)

---

## PART 8: SUMMARY TABLE

| Component | Feature | Status | Evidence |
|-----------|---------|--------|----------|
| **VLM Prompt** | Lists SUBQUESTION role | ✓ Yes | Line 234 |
| **VLM Prompt** | Lists subquestion_of relationship | ✓ Yes | Line 236 |
| **VLM Prompt** | Teaches subquestion semantics | ✗ No | Relies on VLM training |
| **Schema** | SUBQUESTION role type | ✓ Yes | schemas.py:610 |
| **Schema** | subquestion_of relationship type | ✓ Yes | schemas.py:628 |
| **Schema** | parent_question_id field | ✓ Yes | schemas.py:65 |
| **Schema** | VLMStructureItem.role | ✓ Yes | schemas.py:874 |
| **Schema** | VLMRelationshipItem.relationship_type | ✓ Yes | schemas.py:896 |
| **Parser** | Accepts SUBQUESTION role | ✓ Yes | vision_provider.py:505 |
| **Parser** | Accepts subquestion_of relationship | ✓ Yes | vision_provider.py:526 |
| **Parser** | Validates region_ids | ✓ Yes | vision_provider.py:493 |
| **Parser** | Accepts bbox-only structures | ✓ Yes | vision_provider.py:510 |
| **Graph Construction** | Creates SUBQUESTION nodes | ✓ Yes | doc_service.py:622 |
| **Graph Construction** | Creates subquestion_of edges | ✓ Yes | doc_service.py:630 |
| **Graph Construction** | Applies VLM relationships | ✓ Yes | doc_service.py:586 |
| **Extraction** | Walks subquestion_of edges | ✓ Yes | extraction_service.py:482 |
| **Extraction** | Creates parent_question_id | ✓ Yes | extraction_service.py:508 |
| **Extraction** | Validates invariants | ✓ Yes | extraction_service.py:355-368 |
| **MCQ vs Sub** | Deterministic distinction | ⚠️ Fragile | extraction_service.py:901-927 |
| **MCQ vs Sub** | VLM-based distinction | ✓ Capable | Depends on VLM |
| **CRITICAL** | Infer relationships from roles | ✗ Missing | See Part 7 |

---

## PART 9: FINAL VERDICT

### Architecture Capability: ✓ COMPLETE

**Can the pipeline represent Q1 + 1(a)-1(j)?**

**Answer: YES, IF VLM provides the required contract:**

```
REQUIRED CONTRACT:
  1. VLM returns role="SUBQUESTION" for subquestion nodes
  2. VLM returns role="QUESTION" for Q1 node  
  3. VLM returns relationships with:
     - source_ids: [subquestion_node_id]
     - target_ids: [q1_node_id]
     - relationship_type: "subquestion_of"
```

**IF contract fulfilled:**
- ✓ Parser creates correct structures
- ✓ Graph builds correct nodes and edges
- ✓ Extraction creates Questions with parent_question_id
- ✓ Hierarchy preserved end-to-end

### Production Readiness: ⚠️ CONDITIONAL

**Fix #2 is NOT yet production-ready for Q1 subquestions because:**

1. **Verification:** Has only been tested on synthetic 2-block qp.png, not real Q1+10-subquestion PDF
2. **Robustness:** No fallback if VLM returns roles but no relationships
3. **Semantics:** MCQ vs subquestion distinction is fragile without VLM

**Before declaring production-ready:**

1. ✓ Test on real PDF with Q1 + 1(a)-1(j) structure
2. ✓ Verify VLM returns both roles AND relationships
3. ✓ Verify graph contains subquestion_of edges
4. ✓ Verify extraction creates proper hierarchy
5. ⚠️ Optional: Add defensive relationship inference

### Code Change Required?

**Yes, but scope depends on testing results:**

- **If VLM provides relationships:** No change needed (only verification needed)
- **If VLM provides only roles:** Must add defensive relationship inference (small change)
- **For production safety:** Should add defensive inference anyway (recommended)

---

## APPENDIX: REFERENCE LOCATIONS

### Schema Definitions
- DocumentRegionType: `schemas.py:610`
- RelationshipType: `schemas.py:628`
- VLMStructureItem: `schemas.py:874`
- VLMRelationshipItem: `schemas.py:896`
- Question model: `schemas.py:65`

### VLM Prompt
- Prompt building: `document_vision_provider.py:195-268`
- Role enum: `document_vision_provider.py:234`
- Relationship enum: `document_vision_provider.py:236`

### Parser
- _parse_page_understanding: `document_vision_provider.py:474-553`
- Role validation: `document_vision_provider.py:505`
- Relationship validation: `document_vision_provider.py:526`

### Graph Construction
- _build_structure_graph: `document_understanding_service.py:620-678`
- _apply_vlm_page_understandings: `document_understanding_service.py:330-595`
- Relationship application: `document_understanding_service.py:586`

### Extraction
- _extract_from_graph: `intelligent_question_extraction_service.py:155-388`
- _attach_subquestions: `intelligent_question_extraction_service.py:482-530`
- Invariant validation: `intelligent_question_extraction_service.py:355-368`

### Deterministic Analysis
- _analyze_region_hypotheses: `intelligent_question_extraction_service.py:887-970`
- Subquestion vs MCQ: `intelligent_question_extraction_service.py:901-927`

