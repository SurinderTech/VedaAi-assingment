# VedaAI — AI Assessment Extraction & Answer Mapping

> Upload a question paper and a student's handwritten answer sheet.
> VedaAI extracts, maps, and visually connects every answer to its question.

**Teacher → Upload → Extract → Map → Review**

![VedaAI Demo](./docs/demo.gif)

## 🔄 End-to-End Workflow

```mermaid
flowchart LR
    A[👩‍🏫 Teacher] --> B[📄 Upload Documents]

    B --> C{Document Type}

    C -->|Question Paper| D[Question Extraction]
    C -->|Answer Sheet| E[OCR + Layout Extraction]

    D --> F[Structured Questions]
    E --> G[Structured Answer Blocks]

    F --> H[🧠 Mapping Engine]
    G --> H

    H --> I{Mapping Confidence}

    I -->|High| J[✅ Automatic Mapping]
    I -->|Medium| K[🟡 Review]
    I -->|Ambiguous| L[🤖 LLM Verification]
    I -->|No Match| M[⚪ Unmatched]

    L --> N[Final Mapping]
    J --> N
    K --> N

    N --> O[📊 Assessment Workspace]
    O --> P[📍 Highlighted Answer Region]
