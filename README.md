# Multi-Modal RAG System with Agentic Routing

A production-grade, multi-modal Retrieval-Augmented Generation (RAG) system designed to accurately analyze, cross-reference, and answer user queries over research papers containing dense text, charts, figures, and tables. The system incorporates agentic routing across specialist roles, rigorous citation verification to paper and page numbers, and transparent fallback behavior when information is absent.

## Current Status: Day 1 Baseline
- **Text-Only Baseline Completed:** High-throughput PDF extraction, local token chunking, and local vector indexing across landmark research papers.
- **Provider:** OpenAI API configured with role-based model assignments (defaulting to the `reasoning` role for synthesis).
- **Multi-Modal Components:** Planned for subsequent phases (Days 2–5).

## Architecture (5-Line Overview)
1. **Document Ingestion:** Open-access arXiv papers downloaded and processed per page using PyMuPDF.
2. **Token Chunking:** Text split into ~500 token segments with 50-token overlap using `tiktoken` (`cl100k_base`).
3. **Local Embeddings:** Dense 384-dimensional vector representations generated with `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Storage:** Local `faiss-cpu` vector index persisted under `indexes/faiss_index` for sub-millisecond retrieval.
5. **Generation & Verification:** LangChain retrieval chain fetching top-5 chunks, with response synthesis via OpenAI models.

## Role-Based Model Architecture
The system defines role-based model variables in `.env`:
- `OPENAI_MODEL_ORCHESTRATOR` (`gpt-5.5`): Agent routing and workflow dispatch.
- `OPENAI_MODEL_RETRIEVAL` (`gpt-5-mini`): Query expansion and semantic search.
- `OPENAI_MODEL_EVIDENCE` (`gpt-5-mini`): Extraction of snippets and table values.
- `OPENAI_MODEL_REASONING` (`gpt-5.5`): Deep synthesis and citation mapping (active in Day 1).
- `OPENAI_MODEL_VALIDATION` (`gpt-5-mini`): Faithfulness checking and citation audit.

*Note: For GPT-5 reasoning models, completion tokens are bounded via `max_completion_tokens` without passing `temperature`.*

## Windows PowerShell Setup

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```powershell
Copy-Item .env.example .env
# Edit .env and insert your real OpenAI API key:
# OPENAI_API_KEY=sk-...
```

## Running the Pipeline

### 1. Download Research Papers
Downloads 16 open-access arXiv research papers into `data/papers/`:
```powershell
python -m src.download_papers
```

### 2. Run Ingestion Pipeline
Extracts text, splits into token-bounded chunks, and builds the FAISS vector index:
```powershell
python -m src.ingest
```

### 3. Query the RAG CLI
Ask questions and view answers with paper and page citations:
```powershell
python -m src.ask "What is the architecture of the Transformer model?"
```

### 4. Verify API Connection
Verifies connectivity across all configured roles:
```powershell
python -m src.check_api_connection
```

### 5. Run Unit Tests
```powershell
pytest
```

## Project Layout
```text
.
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── api/
│   └── __init__.py
├── audit_reports/
│   ├── day1_phase1_audit.md
│   └── day1_phase2_implementation.md
├── eval/
│   └── __init__.py
├── models/
│   └── __init__.py
├── src/
│   ├── __init__.py
│   ├── ask.py
│   ├── check_api_connection.py
│   ├── download_papers.py
│   ├── ingest.py
│   ├── paper_ids.txt
│   └── rag.py
└── tests/
    ├── __init__.py
    ├── test_ingestion.py
    └── test_retrieval.py
```

## 7-Day Roadmap
- [x] **Day 1: Clean Repository & Text-Only RAG Baseline** *(In Progress)*
- [ ] **Day 2: Visual Ingestion & Chart/Table Extraction**
- [ ] **Day 3: Multimodal Vector Indexing (CLIP/ColPali)**
- [ ] **Day 4: Hybrid Multimodal Retrieval Fusion**
- [ ] **Day 5: Vision-Language Model Integration**
- [ ] **Day 6: Agentic Routing (Orchestrator, Evidence, Reasoning, Validation)**
- [ ] **Day 7: Comprehensive Evaluation, Benchmarking & Production API**
