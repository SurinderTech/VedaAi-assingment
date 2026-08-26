# VedaAI — AI Assessment Extraction & Answer Mapping

> Upload a question paper and a student's handwritten answer sheet. VedaAI extracts, maps, and visually connects every answer to its question.
>
## 🎯 What Problem Does VedaAI Solve?

Evaluating handwritten answer sheets is often a manual search problem.

A teacher has to repeatedly move between the question paper and answer sheet:

```mermaid
flowchart LR
    A[Question Paper] --> B[Find Question]
    B --> C[Search Answer Sheet]
    C --> D[Find Answer]
    D --> E[Read Answer]
    E --> F[Move to Next Question]
    F --> B
```

This becomes increasingly time-consuming when:

Answers are written out of order
Questions contain sub-parts such as 11(a) and 11(b)
An answer continues across multiple pages
Some questions are unanswered
Handwritten content is difficult to locate
Answers contain equations, diagrams, or mixed content


**Teacher → Upload → Extract → Map → Review**
---



##  Core Engineering Approach

VedaAI is intentionally **not** built as:


PDF → LLM → Magic Answer
---


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
```

---

```mermaid
flowchart TB
    A[Document] --> B[Document Processing]

    B --> C[OCR / Native Text Extraction]

    C --> D[Normalized Document Blocks]

    D --> E[Question Extraction]
    D --> F[Answer Extraction]

    E --> G[Question Objects]
    F --> H[Answer Objects]

    G --> I[Mapping Engine]
    H --> I

    I --> J{Confidence}

    J -->|High| K[Automatic Mapping]
    J -->|Ambiguous| L[LLM Verification]
    J -->|Low| M[Human Review]

    K --> N[Assessment Result]
    L --> N
    M --> N

    N --> O[Interactive Workspace]
```
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

    ME --> Viewer

    LLM --> Gemini
    LLM --> Groq
    LLM --> OpenRouter
```

---

## 🧠 Answer Mapping Strategy

VedaAI does **not** send every question and answer to an LLM.

Instead, the mapping engine uses a layered strategy:

```mermaid
flowchart TD

    A[Question + Answer Candidates] --> B{Explicit Question Number?}

    B -->|Yes| C[🎯 High Confidence Mapping]

    B -->|No| D[TF-IDF Semantic Similarity]

    D --> E{Top Candidates Close?}

    E -->|No| F[Select Best Semantic Match]

    E -->|Yes| G[🤖 LLM Verification]

    G --> H{LLM Confident?}

    H -->|Yes| I[Final Mapping]
    H -->|No| J[⚠️ Review Required]

    C --> I
    F --> I
```

### Why this approach?

- **Fast** — deterministic matching handles obvious cases.
- **Cheap** — LLM calls are only made when necessary.
- **Reliable** — low-confidence matches are not forced.
- **Explainable** — every mapping contains its method and confidence.

---

## 🤖 LLM Provider Fallback

```mermaid
flowchart LR

    A[LLM Verification Request] --> B[Primary Provider]

    B -->|Success| C[✅ Response]

    B -->|429 / 5xx / Timeout| D[Fallback Provider 1]

    D -->|Success| C

    D -->|Transient Failure| E[Fallback Provider 2]

    E -->|Success| C

    E -->|All Providers Failed| F[TF-IDF Semantic Result]

    F --> G[Continue Without LLM]
```

Provider order:

**Gemini → Groq → OpenRouter → Semantic-only fallback**

Invalid API keys and invalid requests are **not retried**.

---

## 📍 Interactive Answer Mapping

When a teacher selects a question, VedaAI identifies the corresponding answer and jumps directly to its location on the student's answer sheet.

![Answer Mapping Demo](./docs/answer-mapping.gif)

---




## 📁 Repository Structure

```text
VedaAI/
│
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── document_processor.py
│   │   │   ├── question_extractor.py
│   │   │   ├── answer_extractor.py
│   │   │   ├── mapping_engine.py
│   │   │   ├── embedding_service.py
│   │   │   └── llm_provider.py
│   │   │
│   │   ├── models/
│   │   └── main.py
│   │
│   └── requirements.txt
│
└── frontend/
    ├── components/
    ├── app/
    ├── public/
    └── package.json
```




---



# 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Next.js | Examiner-facing web application |
| Language | TypeScript | Type-safe frontend development |
| Styling | Tailwind CSS | UI implementation and responsive design |
| Animation | Framer Motion | Processing states and subtle UI transitions |
| Backend | FastAPI | REST API and processing orchestration |
| Language | Python | Document processing and AI pipeline |
| PDF Processing | `pypdf` | Extract text from digital PDFs |
| PDF Rendering | `pdf2image` | Render scanned PDF pages |
| OCR | Tesseract + `pytesseract` | Extract text and bounding boxes |
| Semantic Matching | TF-IDF + Cosine Similarity | Match questions with answers |
| ML Utilities | scikit-learn | Text vectorization and similarity |
| LLM Provider | Gemini | Primary ambiguous-case verification |
| LLM Provider | Groq | Fallback LLM provider |
| LLM Provider | OpenRouter | Additional LLM fallback |
| AI Routing | Custom Provider Router | Reliability and provider fallback |
| Storage | In-memory + temporary files | Assignment-scoped storage |
| Database | None | Not required for current scope |
| Vector Database | None | Not required for single-assessment workflow |
| Deployment | Vercel + Python-compatible host | Production deployment |
| Source Control | GitHub | Version control and submission |
