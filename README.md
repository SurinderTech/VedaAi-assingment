# VedaAI

<div align="center">

# 🧠 VedaAI

### AI-Powered Assessment Understanding & Answer Mapping

**Turn hours of manual answer-sheet searching into an intelligent review workflow.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-VedaAI-black?style=for-the-badge)](YOUR_DEMO_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](YOUR_GITHUB_URL)
[![Next.js](https://img.shields.io/badge/Next.js-TypeScript-black?style=for-the-badge&logo=next.js)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=for-the-badge&logo=fastapi)]()
[![AI](https://img.shields.io/badge/AI-Vision%20%2B%20LLM-6366F1?style=for-the-badge)]()

<br />

> **Upload a question paper + a student's handwritten answer sheet.**
>
> **VedaAI extracts the assessment structure, identifies answers, maps them to questions, and visually takes the examiner to the exact answer region.**

<br />

</div>

---
# What is VedaAI 
VedaAI is an AI-powered assessment assistant that understands question papers and handwritten answer sheets, maps each answer to the correct question, and highlights the exact answer region—helping teachers review assessments faster and more efficiently.




# 🎯 Why VedaAI?

Evaluating handwritten answer sheets is often a **manual search problem**.

A teacher may repeatedly have to:

```text
Find Question
      ↓
Search Answer Sheet
      ↓
Locate Student's Answer
      ↓
Read Answer
      ↓
Navigate to Next Question
      ↓
Repeat
```

This becomes increasingly expensive when evaluating hundreds or thousands of answer sheets.<br>

The problem becomes even harder when:<br>

Students answer questions out of order<br>
Questions contain sub-parts such as 11(a) and 11(b)<br>
Answers continue across multiple pages<br>
Some questions are unanswered<br>
Handwriting is difficult to read<br>
Answers contain diagrams or equations<br>
The question paper contains instructions and non-question text<br>
The document contains multiple columns or complex layouts<br>


## VedaAI's goal

Transform:

Manual Document Searching

into:

AI-Assisted Assessment Navigation



# Core Experience

```mermaid
flowchart TD
    A["👨‍🏫 Teacher"] --> B["📄 Upload Question Paper"]
    A --> C["📝 Upload Student Answer Sheet"]

    B --> D["🧠 Question Paper Understanding"]
    C --> E["🧠 Answer Sheet Understanding"]

    D --> D1["Understand Document Structure"]
    D1 --> D2["Identify Question Blocks"]
    D2 --> D3["Extract Questions"]

    E --> E1["Understand Page & Layout Structure"]
    E1 --> E2["Identify Answer Regions"]
    E2 --> E3["Preserve Text + Geometry + Visual Evidence"]

    D3 --> F["🔗 Question ↔ Answer Mapping"]
    E3 --> F

    F --> G{"Mapping Result"}

    G -->|Mapped| H["✅ Question + Answer Linked"]
    G -->|Unanswered| I["⚠️ Question Has No Answer"]
    G -->|Unmatched| J["❓ Answer Has No Clear Question"]
    G -->|Needs Review| K["🔍 Examiner Review Required"]

    H --> L["📊 Assessment Workspace"]
    I --> L
    J --> L
    K --> L

    L --> M["🎯 Exact Answer Region Highlighting"]
    M --> N["👁️ Examiner Reviews Original Evidence"]

    N --> O["✓ Faster Review"]
    N --> P["✓ Traceable Results"]
    N --> Q["✓ Human-in-the-Loop Assessment"]

    classDef input fill:#eef2ff,stroke:#6366f1,stroke-width:2px
    classDef intelligence fill:#ecfeff,stroke:#0891b2,stroke-width:2px
    classDef mapping fill:#fef3c7,stroke:#d97706,stroke-width:2px
    classDef result fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    classDef review fill:#fdf2f8,stroke:#db2777,stroke-width:2px

    class A,B,C input
    class D,D1,D2,D3,E,E1,E2,E3 intelligence
    class F,G mapping
    class H,I,J,K result
    class L,M,N,O,P,Q review
```

---

## ✨ What VedaAI Can Understand
## 📄 Question Paper Understanding



VedaAI is designed to understand complex question-paper layouts containing:



Structure	Examples
❓ Questions	Regular questions<br>
📝 MCQs	Questions + multiple options<br>
🔹 Options	A / B / C / D<br>
🔢 Subquestions	11(a), 11(b), (i), (ii)<br>
📚 Sections	Section A, B, C...<br>
📌 Instructions	General examination instructions<br>
📊 Tables	Structured tabular information<br>
🖼️ Diagrams	Visual question content<br>
📐 Multi-column layouts	Two-column / mixed layouts<br>
📄 Multi-page questions	Questions spanning pages<br>
🧩 Mixed content	Text + figures + tables<br>

The system is specifically designed to avoid treating every piece of extracted text as a question.

<p align="center">
      <h1>Results<h1>
  <img src="Screenshot (717).png" width="48%">
  <img src="" width="48%">
</p>


