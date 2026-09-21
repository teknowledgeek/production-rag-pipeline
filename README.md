# Production RAG Pipeline

A retrieval-augmented generation (RAG) system built with production concerns in mind — hybrid retrieval, grounded generation, pluggable embedding backends, and real deployment trade-offs — not just a notebook demo.

> 🚧 **Status:** Core pipeline (ingest → embed → retrieve → generate) working end-to-end, exposed via FastAPI. Evaluation suite in progress.

**Live demo:** [https://production-rag-pipeline-onrender-com.onrender.com/docs#/default/query_rag_query_post]
**Blog / build-in-public posts:** [https://www.linkedin.com/posts/activity-7500047064018595840-yhxx?utm_source=share&utm_medium=member_desktop&rcm=ACoAABXW5ZkBHbueEkoDkDc5KjOfgVfmwe6PzMI]

---

## Why this project exists

Most RAG tutorials stop at "embed some chunks, retrieve top-k, ask an LLM." That works in a demo and falls apart in production — retrieval quality is inconsistent, there's no way to measure whether changes actually help, and memory/cost/latency are invisible until they're a problem.

This project builds a RAG pipeline the way I'd want to hand it off to a team: with hybrid retrieval, grounded generation, deployment-aware architecture, and evaluation baked in — applied to generic QnA

## Architecture

```
                                                                 
   Documents  ──▶  Ingestion &   ──▶  Embedding &   ──▶  Vector 
   (PDF/MD/   │    Chunking      │    Metadata     │    Store   
   HTML/TXT)  │  (Recursive      │  (fastembed /   │  (Qdrant)  
              │   splitter)      │   sentence-      │            
              │                  │   transformers)  │            
                                                                 
                                                                       
                                                                       ▼
   User      ◀──  Generation    ◀──  Optional       ◀──  Hybrid     
   Answer     │   (Groq/OpenAI/ │    Cross-Encoder   │   Retrieval   
   + sources  │    Anthropic,   │    Re-ranking      │  (dense+BM25) 
              │    citation-    │  (toggle: RAM      │   + RRF       
              │    forced)      │   dependent)       │   fusion      
                                                                    
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking strategy | Recursive/semantic chunking (`RecursiveCharacterTextSplitter`), not fixed-size | Preserves paragraph/sentence structure before falling back to hard cuts — better retrieval precision than naive fixed-size splitting. **Limitation:** not structure-aware (no heading/table awareness) — a v2 candidate. |
| Embedding backend | Pluggable: `fastembed` (default) or `sentence-transformers`, both serving `BAAI/bge-small-en-v1.5` | `fastembed` is ONNX-quantized with no PyTorch dependency — dramatically lighter memory footprint, critical for free-tier deployment (512MB RAM limits). `sentence-transformers` kept as an option for local dev / higher-RAM environments. This trade-off — and the deployment failures that motivated it — is documented below. |
| Retrieval | Hybrid (dense + BM25) fused via Reciprocal Rank Fusion (RRF), with **optional** cross-encoder re-ranking | Naive top-k cosine similarity misses exact-match/keyword-heavy queries; BM25 catches those. RRF combines rankings rather than raw scores, since dense (cosine) and BM25 scores aren't on the same scale. Re-ranking is gated behind `ENABLE_RERANKING` since the cross-encoder requires loading a second PyTorch model — a real memory cost, made explicit rather than hidden. |
| Vector store | Qdrant Cloud (free tier) | Open-source, production-capable, avoids vendor lock-in for a portfolio piece; cloud-hosted for persistence across deploys. |
| Grounding | Forced citation + explicit "I don't have enough information" fallback in the system prompt | Reduces hallucination risk — a core production concern, not just a demo nicety. Verified by testing with out-of-corpus questions. |
| LLM provider | Pluggable: Groq (default, free tier) / OpenAI / Anthropic | Groq's free tier (no credit card, generous rate limits) removed a hard cost dependency during development. Provider is a one-line `.env` swap, not a code change. |
| Model hosting | FastAPI with models loaded once at startup (`lifespan` context manager), not per-request | Loading embedding/re-ranking models takes real time — doing it once at startup instead of per-request is the difference between a usable API and a 10+ second-per-query one. |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision/recall) — *in progress* | Makes "did this change help?" answerable with numbers, not vibes. |

## Deployment story (a real trade-off, not a footnote)

This project was deployed, broke, and was fixed — documented honestly rather than only showing the end state:

1. **Render (free tier)** — first deployment target. Hit a hard `512MB RAM` ceiling: loading `sentence-transformers` (full PyTorch) plus the cross-encoder re-ranker exceeded it, causing repeated OOM kills during startup, even after disabling re-ranking alone.
2. **Hugging Face Spaces** — investigated as an alternative (16GB free CPU tier), but HF changed its pricing policy: Docker/Gradio Spaces now require a paid plan; only static (non-compute) Spaces are free.
3. **Root-cause fix: swapped `sentence-transformers` for `fastembed`** (ONNX-quantized, no PyTorch dependency) as the default embedding backend, cutting the memory footprint enough to run on Render's genuinely free tier. `sentence-transformers` remains available as an opt-in backend for local development or higher-RAM environments.

This is documented here deliberately: understanding *why* something didn't fit a resource constraint, and fixing the actual bottleneck rather than just paying for more RAM, is the kind of decision this README is meant to make visible.


## Tech stack

- **Orchestration:** Python, LangChain (`langchain-community`, `langchain-text-splitters`)
- **Document loading:** `PyPDFLoader`, `UnstructuredMarkdownLoader`, `UnstructuredHTMLLoader`, `TextLoader`
- **Embeddings:** `fastembed` (default) or `sentence-transformers` — both serving `BAAI/bge-small-en-v1.5`
- **Sparse retrieval:** `rank-bm25`
- **Re-ranking (optional):** `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
- **Vector store:** Qdrant Cloud
- **LLM:** Groq (default) / OpenAI / Anthropic — swappable via `.env`
- **API:** FastAPI + Uvicorn, models loaded once at startup via `lifespan`
- **Evaluation:** RAGAS *(in progress)*
- **Deployment:** Docker + Render
- **CI:** GitHub Actions (lint + test on every PR)

## Getting started

```bash
# clone
git clone https://github.com/<your-username>/production-rag-pipeline.git
cd production-rag-pipeline

# environment
python -m venv venv
venv\Scripts\Activate.ps1     # Windows PowerShell
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt

# configure
cp .env.example .env   # add your API keys and Qdrant credentials
```

`.env` variables:
```
QDRANT_URL=
QDRANT_API_KEY=
EMBEDDING_BACKEND=fastembed        # or: sentence-transformers
ENABLE_RERANKING=true              # set false on memory-constrained deployments
LLM_PROVIDER=groq                  # or: openai / anthropic
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Run the pipeline:
```bash
# 1. Ingest and chunk documents
python -m src.ingest --source ./data/docs --output ./data/chunks.json

# 2. Embed and index into Qdrant
python -m src.embed --input ./data/chunks.json --collection rag_docs

# 3. Test retrieval standalone
python -m src.retrieve --query "your question" --top-k 5

# 4. Test full retrieve + generate standalone
python -m src.generate --query "your question"

# 5. Run the API
uvicorn src.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API UI.

Or with Docker:
```bash
docker build -t rag-pipeline .
docker run -p 8000:8000 --env-file .env rag-pipeline
```

## Project structure

```
production-rag-pipeline/
├── src/
│   ├── ingest.py           # document loading, cleaning, chunking
│   ├── embed.py             # pluggable embedding + Qdrant indexing
│   ├── retrieve.py          # hybrid retrieval (dense + BM25) + optional re-ranking
│   ├── generate.py          # grounded generation, citation-forced prompting
│   └── api.py                 # FastAPI app, startup-loaded retriever
├── eval/
│   ├── eval_set.json        # question/answer pairs for evaluation
│   └── run_eval.py           # RAGAS evaluation runner
├── tests/
├── data/
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Core ingestion + chunking pipeline
- [x] Pluggable embedding backend (fastembed / sentence-transformers) + Qdrant indexing
- [x] Hybrid retrieval (dense + BM25 + RRF fusion)
- [x] Optional cross-encoder re-ranking (memory-aware toggle)
- [x] Grounded generation with forced citations, multi-provider LLM support
- [x] FastAPI backend with startup-loaded models
- [x] Dockerized, deployed (Render free tier)
- [ ] Evaluation suite (RAGAS) with baseline numbers — with/without re-ranking comparison
- [ ] Streamlit frontend
- [ ] Logging, caching, cost tracking
- [ ] Write companion blog post on the deployment/memory trade-off in depth

## License

MIT
