
## ✦ Built at SDA Himmah Hackathon 2026

> Participated — Modrik Team
---

<div align="center">

```
╔══════════════════════════════════════════╗
║                                          ║
║               M O D R I K                ║
║                   مدرك                   ║
║                                          ║
║   Every worker deserves to know their    ║
║                 rights.                  ║
║                                          ║
╚══════════════════════════════════════════╝
```

**An Arabic-first AI legal assistant for Saudi Labor Law.**
Ask in plain Arabic. Upload your contract. Get an answer cited by article.

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Local%20RAG-009688?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

## ✦ What is Modrik?

Saudi Labor Law spans hundreds of articles across three states — active, amended, and repealed — and most workers never open it until they're already in a dispute.

**Modrik (مدرك)** lets anyone ask about their rights in plain Arabic, upload their actual contract or case file, and get a personal answer grounded in the specific article that applies — not a generic summary.

> *"هل عقدي يطابق نظام العمل؟"*
> Modrik reads your contract, retrieves the relevant articles — whether still in force, amended, or repealed — and tells you exactly where you stand, citing the law instead of guessing.

---

## ✦ How a question gets answered

```
        USER ASKS A QUESTION  (+ optional contract / case file)
                        │
                        ▼
              ┌───────────────────┐
              │     PII GUARD     │   redacts Saudi ID / IBAN
              │  (regex)          │   before anything is sent
              └─────────┬─────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  UPLOADED FILE                  FAISS RETRIEVAL
  (txt / pdf extracted)     query embedded with
                          multilingual-e5-large
                           → top-k matching Labor
                             Law articles, local
        └───────────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │   GPT-4o-mini     │   streamed response,
              │   (streamed)      │   cites article numbers,
              └─────────┬─────────┘   flags amended/repealed
                        ▼
              ANSWER + SOURCES POPOVER
              (article matches with similarity %)
```

| Step | What happens |
|------|---------------|
| **Input** | User types in Arabic (or English) and may attach a `.txt`/`.pdf` contract or case document |
| **Privacy guard** | Saudi National ID/Iqama numbers and IBANs are regex-redacted before anything leaves the session |
| **Retrieval** | The query is embedded locally with `multilingual-e5-large` and FAISS searches the Labor Law index (active + amended + repealed) for the top-k matching articles |
| **Generation** | GPT-4o-mini streams a personalized answer, always citing the article number and flagging amended/repealed status |
| **Transparency** | A sources popover shows the exact retrieved articles with similarity scores, plus the contract excerpt actually used |

---

## ✦ Tech stack

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND / UI                    │
│        Streamlit · RTL Arabic chat interface        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                        core/                        │
│                                                     │
│  ┌────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │ pii_guard  │  │ file_service  │  │ vector_store│ │
│  │ regex      │  │ PyMuPDF       │  │ FAISS,      │ │
│  │ redaction  │  │ txt / pdf     │  │ local, free │ │
│  └────────────┘  └───────────────┘  └──────┬──────┘ │
│                                           │         │
│                                   ┌───────▼───────┐ │
│                                   │  ai_service   │ │
│                                   │  OpenAI stream│ │
│                                   └───────────────┘ │
└─────────────────────────────────────────────────────┘
```

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini, streamed token-by-token |
| Embeddings | `intfloat/multilingual-e5-large`, run locally (see below) |
| Vector store | FAISS (`IndexFlatIP` over normalized vectors) |
| PDF / text extraction | PyMuPDF (`fitz`) |
| Law ingestion | pandas + openpyxl — Excel → smart-chunked articles |
| Privacy | regex-based PII redaction (Saudi ID/Iqama, IBAN) |
| UI | Streamlit, RTL layout, custom CSS, IBM Plex Sans Arabic |
| Deployment | Docker / docker-compose, Hugging Face Spaces |

---

## ✦ Why this embedding model

Most RAG demos default straight to an embedding API. Modrik instead runs `multilingual-e5-large` **locally** — the multilingual E5 model from **Microsoft Research** — picked after comparing leading multilingual embedding models against published Arabic retrieval benchmarks (MIRACL, MTEB), where it consistently ranks among the strongest open models for Arabic semantic search. That matters here specifically: Saudi Labor Law articles are dense, formal legal Arabic, not casual text.

The practical upside: embedding has zero per-query API cost, and the index — active, amended, and repealed articles — is pre-warmed at container startup, so retrieval is instant from the first question.

---

## ✦ Smart law ingestion

Saudi Labor Law isn't one flat document — articles get amended or repealed, and the old text still matters for context. `scripts/ingest_labor_law.py` ingests three separate sources and tags each one:

| Source | Tag shown to the model |
|--------|------------------------|
| `labor_law_active.xlsx` | ✅ نظام نشط وسارٍ حالياً |
| `labor_law_amended.xlsx` | ⚠️ هذه المادة معدّلة |
| `labor_law_repealed.xlsx` | 🚫 هذه المادة ملغاة ولا تُطبَّق حالياً |

Articles longer than 600 characters are split on definition markers or paragraph breaks first, then merged back up to the chunk limit — so no sub-chunk ever loses its article header or keyword context.

---

## ✦ Project structure

```
Modrik/
│
├── app.py                      ← Streamlit entry point (chat UI, RTL Arabic)
├── core/
│   ├── ai_service.py           ← OpenAI streaming + system prompt
│   ├── file_service.py         ← txt/pdf extraction + chunking
│   ├── pii_guard.py            ← Saudi ID / IBAN redaction
│   └── vector_store.py         ← FAISS index, embed/retrieve
├── scripts/
│   └── ingest_labor_law.py     ← builds the FAISS index from Excel sources
├── tests/
│   └── test_pipeline.py        ← full unit test suite
├── data/                       ← active / amended / repealed Labor Law (Excel)
├── index/                      ← FAISS index + chunks (generated)
├── assets/                     ← logo
├── Dockerfile / docker-compose.yml / entrypoint.sh
└── requirements.txt
```

---

## ✦ Tests

Every public function ships with a test: chunking edge cases, `.txt`/`.pdf` extraction (PDF mocked via `fitz`), the OpenAI streaming contract, vector store retrieval, and every PII redaction pattern.

```bash
python -m pytest tests/ -v
```

---

## ✦ Privacy by default

Before any text — typed or uploaded — reaches OpenAI or gets logged, `core/pii_guard.py` strips Saudi National ID/Iqama numbers and IBANs with a regex pass, replaces them with `[رقم هوية]` / `[رقم IBAN]`, and tells the user how many were redacted. No opt-in required, no configuration needed.

---

## ✦ Quick start

### Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- An OpenAI API key

### 1 — Copy the env file and add your key

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
OPENAI_API_KEY=sk-...
```

### 2 — Build and start

```bash
docker-compose up --build
```

On first run, the FAISS index is built automatically (~1-2 minutes). Subsequent starts are instant.

### 3 — Open in browser

```
http://localhost:8501
```

### SSL error on corporate / university networks

```
SSLError: certificate verify failed
```

This is caused by a firewall or SSL-inspecting proxy. The fix is already included in the `Dockerfile` — no extra steps needed.

### Useful commands

```bash
# Stream logs
docker-compose logs -f

# Stop the project
docker-compose down

# Rebuild after code or requirements.txt changes
docker-compose up --build

# Run the test suite
python -m pytest tests/ -v
```

---

## ✦ Roadmap

| Status | Feature |
|--------|---------|
| ✅ | RAG over active / amended / repealed Labor Law articles |
| ✅ | Contract & case file upload (`.txt`/`.pdf`) with personalized analysis |
| ✅ | Automatic PII redaction (Saudi ID, IBAN) |
| ✅ | Source transparency — article + similarity score popover |
| ✅ | Full automated test suite |
| ✅ | Dockerized, one-command run |
| 🔜 | More legal domains beyond Labor Law |
| 🔜 | Multi-turn case memory across sessions |
| 🔜 | Export consultation as PDF |

---

<div align="center">

*Every worker deserves to know where they stand.*

</div>
