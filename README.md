# VedaAI

### AI-Powered Assessment Understanding & Answer Mapping

> Upload a question paper and a student's handwritten answer sheet.  
> VedaAI understands the documents, extracts their structure, maps answers to questions, and lets examiners inspect the exact answer regions.

[![Next.js](https://img.shields.io/badge/Next.js-TypeScript-black)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688)]()
[![AI](https://img.shields.io/badge/AI-Vision%20%2B%20LLM-blue)]()
[![Status](https://img.shields.io/badge/Status-Assignment%20Demo-orange)]()

---

## Overview

Evaluating handwritten answer sheets is still a highly manual process.

For every question, a teacher or examiner may need to:

1. Find the question in the question paper.
2. Search through the student's answer sheet.
3. Locate the corresponding answer.
4. Determine whether the answer belongs to that question.
5. Read the handwritten response.
6. Navigate across pages when the answer continues.
7. Handle unanswered or out-of-order questions.
8. Repeat the process for the entire paper.

This becomes extremely expensive in time when an examiner has to evaluate hundreds or thousands of answer sheets.

**VedaAI explores how multimodal AI and document intelligence can reduce this repetitive workload.**

Instead of treating a document as a flat collection of OCR text, VedaAI attempts to understand its visual and semantic structure.

---

# The Core Problem

The challenge is not simply:

```text
PDF → OCR → Text

The real problem is:

Question Paper
      ↓
Understand document structure
      ↓
Extract questions
      ↓
Understand answer sheet
      ↓
Extract answer regions
      ↓
Determine which answer belongs to which question
      ↓
Preserve the exact visual evidence
      ↓
Help the examiner review the result

VedaAI is built around this workflow.

Core Experience
Teacher
   │
   ▼
Upload Question Paper
   │
   ▼
Upload Student Answer Sheet
   │
   ▼
Document Understanding
   │
   ├── Question Structure
   └── Answer Structure
   │
   ▼
Question ↔ Answer Mapping
   │
   ├── Mapped
   ├── Unanswered
   ├── Unmatched
   └── Needs Review
   │
   ▼
Assessment Workspace
   │
   └── Exact Answer Region Highlighting
What VedaAI Does
Question Paper Understanding
```

VedaAI can analyze complex question-paper layouts containing:

Questions
MCQs
Options
Subquestions
Sections
Instructions
Tables
Diagrams
Headers and footers
Multi-column layouts
Multi-page questions
Other document structures

The system is designed to avoid treating every piece of extracted text as a question.

Student Answer Understanding

The answer-sheet pipeline is designed to identify:

Answer regions
Handwritten content
Multi-line answers
Multi-page answers
Answers written out of order
Unanswered questions
Answers that cannot be confidently matched
Answer Mapping

VedaAI attempts to determine:

Question 7
      ↓
Student Answer
      ↓
Page 4
      ↓
Exact answer region

When confidence is insufficient, the result can be surfaced for examiner review instead of silently forcing an incorrect mapping.

Visual Answer Highlighting

When an examiner selects a question, VedaAI can navigate to the associated answer region.

This creates a direct relationship:

Question
   ↓
Mapped Answer
   ↓
Answer Page
   ↓
Answer Region

The examiner does not have to manually search the entire answer sheet.

Vision-First Document Intelligence

A major design decision in VedaAI is that document understanding should not depend primarily on:

fixed PDF templates
hardcoded coordinates
document-specific rules
question-number assumptions
keyword-only classification

Instead, the intended architecture is:

Rendered Page Image
        ↓
Multimodal Vision Model
        ↓
Visual / Semantic Structures
        ↓
Geometry-Based OCR Grounding
        ↓
Document Structure Graph
        ↓
Question / Answer Extraction

The vision model provides semantic understanding of the page while OCR/native PDF extraction provides exact textual evidence.

This separation is important because visual models are good at understanding layout and relationships, while OCR is useful for preserving exact text and coordinates.

Document Understanding Architecture
Why a Document Structure Graph?

A document is not simply a list of text blocks.

For example:

Section A
   │
   ├── Question 1
   │      ├── Option A
   │      ├── Option B
   │      ├── Option C
   │      └── Option D
   │
   └── Question 2
          ├── Option A
          ├── Option B
          ├── Option C
          └── Option D

Representing these relationships explicitly makes downstream extraction significantly more reliable.

The graph can represent relationships such as:

option_of
subquestion_of
section_member
continuation_of
contains
belongs_to
follows
associated_visual

The graph also preserves provenance and confidence information.

Semantic Uncertainty

Real-world documents are not always perfectly readable.

VedaAI therefore treats uncertainty as a first-class concept.

Possible semantic states include:

CONFIDENT
PARTIAL
AMBIGUOUS
UNKNOWN
UNRESOLVED
CONFLICTING

Instead of silently converting uncertain interpretations into facts, the pipeline can preserve uncertainty for downstream review.

This is particularly important for:

poor scans
complex layouts
ambiguous answer mappings
handwritten responses
incomplete model responses
unusual document structures
Question Extraction

Extracted questions retain structural information such as:

{
  "question_number": "11(a)",
  "question_type": "SHORT_ANSWER",
  "question_text": "...",
  "page_number": 4,
  "confidence": 0.96,
  "options": [],
  "source_region_ids": []
}

For MCQs:

{
  "question_number": "7",
  "question_type": "MCQ",
  "question_text": "...",
  "options": [
    {
      "label": "A",
      "text": "...",
      "confidence": 0.95
    },
    {
      "label": "B",
      "text": "...",
      "confidence": 0.96
    }
  ]
}

The actual API structure may contain additional provenance and semantic fields.

Answer Mapping

VedaAI uses structured document information and matching signals to associate answers with questions.

Conceptually:

The important principle is:

Do not force a mapping when the evidence is insufficient.

Handling Real Examination Documents

The system has been tested against a complex real-world examination document containing:

multiple pages
sections
instructions
MCQs
multiple options
subquestions
descriptive questions
page transitions
non-question document content
dense page layouts

One of the development goals was specifically to prevent unrelated document text from becoming false questions.

For example:

Document instruction
       ↓
should NOT become
       ↓
Question 37

Likewise:

Option A
Option B
Option C
Option D

should remain associated with their actual parent question rather than being absorbed into unrelated question text.

Handling Question Structure

VedaAI is designed around semantic structure rather than one fixed numbering convention.

Examples include:

1
2
3
11(a)
11(b)
12(i)
12(ii)

Subquestions can be represented as independent structured entities while retaining their relationship with the parent question.

Handling Complex Layouts

The document understanding layer is designed to work with layouts such as:

Single Column
     ↓
Multi Column
     ↓
Tables
     ↓
Diagrams
     ↓
Mixed Text + Visuals
     ↓
Question + Options
     ↓
Question + Subquestions

The system uses page geometry and visual relationships rather than assuming a fixed page layout.

LLM Provider Architecture

VedaAI uses a provider abstraction so that AI capabilities are not tightly coupled to a single model vendor.

Conceptually:

This allows provider/model configuration to change without redesigning the document pipeline.

VedaAI Assistant

VedaAI also includes an in-product AI Assistant designed to help the examiner understand the current assessment.

The Assistant can provide context-aware help around:

extracted questions
student answers
unanswered questions
answer mappings
review items
scores
assessment summaries
grading information
assessment insights

The Assistant is designed as an interface to VedaAI's assessment data rather than as a generic chatbot.

Technology Stack
Layer	Technology
Frontend	Next.js
Frontend Language	TypeScript
Styling	Tailwind CSS
UI Animation	Framer Motion
Backend	FastAPI
Backend Language	Python
PDF Processing	pypdf / PDF rendering
OCR	OCR engine + native PDF text extraction
Computer Vision	Multimodal Vision Model
Semantic Understanding	LLM / VLM
Document Structure	Semantic Graph
Text Similarity	scikit-learn / semantic matching
AI Routing	Custom Provider Router
Storage	Temporary / in-memory storage
Database	Not required for current assignment scope
Deployment	Vercel + Python-compatible backend hosting
Version Control	GitHub
Project Structure
VedaAI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │
│   │   ├── models/
│   │   │
│   │   ├── schemas/
│   │   │
│   │   └── services/
│   │       ├── document_processor.py
│   │       ├── document_vision_provider.py
│   │       ├── document_understanding_service.py
│   │       ├── intelligent_question_extraction_service.py
│   │       ├── question_extractor.py
│   │       ├── answer_extractor.py
│   │       ├── mapping_engine.py
│   │       ├── assessment_result_service.py
│   │       ├── embedding_service.py
│   │       └── llm_provider.py
│   │
│   ├── tests/
│   │
│   ├── requirements.txt
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── types/
│   │   └── ...
│   │
│   └── package.json
│
├── docs/
│
└── README.md
Local Development
Prerequisites

Make sure you have:

Node.js
Python 3.10+
Git
Required AI provider API credentials
Backend
cd backend

python -m venv venv
Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create your environment file:

backend/.env

Configure the required provider credentials.

Then start FastAPI:

python -m uvicorn app.main:app --reload

The API will normally be available at:

http://localhost:8000

Interactive API documentation:

http://localhost:8000/docs
Frontend
cd frontend
npm install
npm run dev

The frontend will normally be available at:

http://localhost:3000
Environment Variables

API credentials should never be committed to Git.

Example:

GEMINI_API_KEY=your_key_here

OPENROUTER_API_KEY=your_key_here

OPENROUTER_ASSISTANT_API_KEY=your_assistant_key_here

Use the actual variables required by the current backend configuration.

Never commit .env files containing real credentials.

Testing

Run the backend test suite:

cd backend
python -m pytest tests/ -v

The project includes regression coverage for document understanding and extraction behavior, including:

semantic completeness
VLM failures
graph integrity
question deduplication
option relationships
cross-page continuation
decomposition
API structural preservation
Design Principles

VedaAI follows several important engineering principles.

1. Vision before assumptions

The system should understand the visual document rather than assuming a predefined template.

2. OCR as evidence

OCR/native extraction provides textual evidence and geometry.

It should not be treated as the complete semantic interpretation of a page.

3. Relationships matter

A document is a structured visual object.

The relationship between:

Question → Option
Question → Subquestion
Question → Continuation
Answer → Region

is as important as the text itself.

4. Preserve provenance

Extracted information should retain information about where it came from whenever possible:

Page
Bounding Box
Region
Confidence
Source
5. Never silently convert uncertainty into truth

When evidence is insufficient, the system should be able to surface the result for review.

6. Keep the architecture document-agnostic

The goal is not to create a parser for one examination board.

The goal is to build a reusable document-understanding foundation.

Current Scope

The current implementation focuses on the assignment's core workflow:

Question Extraction
        ↓
Answer Extraction
        ↓
Answer Mapping
        ↓
Visual Answer Location
        ↓
Assessment Review

Optional grading and AI insights can be layered on top of this pipeline.

Known Limitations

Real-world document intelligence is difficult.

Accuracy can vary depending on:

scan quality
image resolution
handwriting quality
unusual layouts
diagrams
tables
document complexity
model availability
OCR quality
ambiguous answer placement

The current implementation should therefore be treated as an AI-assisted system rather than a replacement for examiner judgment.

The system intentionally allows uncertain cases to remain reviewable.

Why This Project Matters

The interesting part of this problem is not simply extracting text from a PDF.

The real challenge is understanding the relationship between two different visual documents:

              QUESTION PAPER
                    │
                    │
             "What was asked?"
                    │
                    ▼
               QUESTION
                    │
                    │
              "Who answered?"
                    │
                    ▼
              ANSWER SHEET
                    │
                    │
              "Where is it?"
                    │
                    ▼
             ANSWER REGION

A teacher should not have to manually search an entire handwritten answer sheet every time they want to inspect one question.

VedaAI attempts to turn that search problem into an intelligent navigation problem.

Future Direction

The current assignment implementation provides a foundation for a much larger assessment intelligence platform.

Potential future capabilities include:

multi-student assessment processing
batch answer-sheet evaluation
stronger handwriting understanding
diagram-aware answer evaluation
mathematical expression understanding
richer grading assistance
teacher feedback generation
performance analytics
topic-level insights
assessment comparison
human-in-the-loop review workflows
long-term assessment history
institutional integrations

The long-term vision is to make assessment evaluation significantly less repetitive while keeping the examiner in control.

Assignment Context

This project was developed as part of an AI Assessment Extraction & Answer Mapping assignment.

The assignment requires:

Question paper upload
Student answer-sheet upload
Question extraction
Answer extraction
Question-answer mapping
Correct printed ordering
Subquestion handling
Out-of-order answers
Unanswered questions
Unmatched answers
Exact answer-region highlighting
Multi-page answer handling

VedaAI was built around these requirements while exploring a broader vision-first approach to document intelligence.

Author

Surinder Kumar

B.Tech — Computer Science & Engineering

Final Thought

VedaAI is an exploration of what happens when document processing moves beyond:

"Extract the text."

toward:

"Understand what this document means,
how its parts relate to each other,
and show the examiner the evidence."

### One thing I'd change from your current README

I would **remove this claim** from the old version:

> `Semantic Matching | TF-IDF + Cosine Similarity`

if your current implementation has moved substantially toward the **vision/graph-based architecture** we've been working on. Otherwise an evaluator may look at the README, inspect the code, and wonder why the README describes an older architecture.

Likewise, don't advertise **“Gemini → Groq → OpenRouter”** as a fixed provider order unless that is genuinely the current production implementation. Your README should describe what the code **actually does today**, not what an earlier version did.

This version presents VedaAI as a serious engineering project without making the dangerous claim that it is
