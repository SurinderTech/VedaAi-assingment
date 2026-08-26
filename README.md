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
---

That immediately communicates what the entire system does.

---

### 3. Architecture

```markdown
## 🏗️ System Architecture

```mermaid
flowchart TB

    subgraph Frontend["Frontend — Next.js"]
        UI[Teacher Workspace]
        Upload[Upload Interface]
        Viewer[Answer Sheet Viewer]
        UI --> Upload
        UI --> Viewer
    end

    subgraph Backend["Backend — FastAPI"]

        API[Assessment API]

        DP[Document Processor]
        QE[Question Extractor]
        AE[Answer Extractor]
        ME[Mapping Engine]

        SIM[TF-IDF + Cosine Similarity]
        LLM[LLM Provider Router]

        API --> DP
        DP --> QE
        DP --> AE

        QE --> ME
        AE --> ME

        ME --> SIM
        ME --> LLM
    end

    subgraph Providers["External AI Providers"]
        Gemini[Gemini]
        Groq[Groq]
        OpenRouter[OpenRouter]
    end

    Upload --> API
    API --> UI

    LLM --> Gemini
    LLM --> Groq
    LLM --> OpenRouter

    ME --> Viewer

    N --> O[📊 Assessment Workspace]
    O --> P[📍 Highlighted Answer Region]
