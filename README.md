---
title: PDF-Constrained RAG Agent
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-upgrade
short_description: Chat with any PDF, strict grounding, citations on every claim
---

# PDF-Constrained Conversational RAG Agent

**Task 3** — chat with any PDF, with **strict grounding**, **citations on every claim**, and **a fixed refusal string** for out-of-scope queries.

This repo is the Phase 1 + Phase 2 implementation of an architecture that scales up to multimodal + propositions + KG (Phase 3). Phase 2 adds:
- **RAPTOR tree** (UMAP+GMM clustering with Claude-generated cluster/section/doc summaries) so holistic queries like "summarize main contributions" actually work
- **Anthropic Contextual Retrieval headers** (with prompt caching of the full doc body) — up to 67% retrieval-error reduction
- **Deep mode**: HyDE + multi-query + step-back + query decomposition + two-pass verifier (drops unsupported claims) — toggle in the UI
- **Async ingestion worker** (arq + Redis): `/upload` returns immediately with a `doc_id`; UI polls `/upload/status/{doc_id}` and shows phase-by-phase progress
- **Chainlit UI** alongside Streamlit, with chat-profile switching between instant and deep modes

> **Refusal string** (must match exactly): `I cannot answer this from the provided document.`

---

## Phase 1 stack

| Component | Choice | Why |
|---|---|---|
| Layout / table parsing | **Docling 2.x** | Best open-source layout AI; multi-column reading order; TableFormer for tables |
| Embeddings | **BGE-M3** (BAAI/bge-m3) | 1024-dim dense + sparse in one model; 100+ languages incl. Hindi/Tamil/Telugu/Bengali/Marathi/Gujarati/Kannada/Malayalam/Punjabi |
| Vector DB | **Qdrant 1.12** | Native hybrid (dense + sparse) with RRF fusion via Query API |
| Reranker | **bge-reranker-v2-m3** | Multilingual cross-encoder, ~560 MB |
| Generation | **Claude Sonnet 4.5** + Anthropic **Citations API** (custom-content mode) | Sentence-level grounded citations natively |
| Faithfulness gate | **Claude Haiku 4.5** LLM-judge | Phase 1 substitute for bespoke-minicheck-7B |
| OOS detection | cosine vs document centroid + retrieval-score floor | Two-stage refusal |
| API / Frontend | **FastAPI** + **Streamlit** | Production-style backend, demo-grade UI |

**Multilingual**: BGE-M3 covers all popular world languages (English, Spanish, French, German, Chinese, Japanese, Arabic) plus the listed Indian regionals out of the box.

Total model downloads on first run: ~1.8 GB (BGE-M3 ~1.2 GB + bge-reranker-v2-m3 ~560 MB). Cached under `HF_HOME` (defaults to `./.cache/huggingface`).

---

## Quickstart

> **Reviewing this submission?** See **[SUBMISSION.md](SUBMISSION.md)** for the deployed URL, the test queries, and what each architecture choice signals. The rest of this README is for running it locally.

---

## Deploying to Hugging Face Spaces

This repo ships a single-container Docker image that runs everything (Qdrant + Redis + FastAPI + arq worker + Chainlit) under supervisord. Deploy in 4 steps:

1. **Push the repo to a new HF Space** with `sdk: docker` (the YAML at the top of this file is already configured).
   ```bash
   huggingface-cli login
   huggingface-cli repo create pdf-rag --type space --space_sdk docker
   git remote add hf https://huggingface.co/spaces/<your-username>/pdf-rag
   git push hf main
   ```
2. **Set Secrets** in the Space → Settings → Variables and secrets:
   - `ANTHROPIC_API_KEY` (Secret) — your `sk-ant-…` key
   - `ANTHROPIC_GENERATION_MODEL=claude-opus-4-7` (Variable, optional)
   - `ANTHROPIC_JUDGE_MODEL=claude-sonnet-4-6` (Variable, optional)
3. **Wait for the build** (~10 min first time — pre-downloads BGE-M3 + reranker so cold starts are fast).
4. **Set a spending cap** at console.anthropic.com → Settings → Limits → $10/mo. Belt and suspenders.

Public URL is `https://huggingface.co/spaces/<your-username>/pdf-rag`. Reviewers click and chat — no setup.

---

## Running locally

### 1. Prerequisites

- Python 3.12
- Docker (for Qdrant)
- An Anthropic API key
- ~4 GB free disk for Qdrant storage + HF model cache

### 2. Install

```bash
# From project root
cp .env.example .env
# Edit .env to set ANTHROPIC_API_KEY

# Recommended: uv (fast)
uv sync

# Or pip:
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 3. Start Qdrant + Redis

```bash
docker compose up -d qdrant redis
# Qdrant dashboard: http://localhost:6333/dashboard
# Redis is used by the async ingestion worker.
```

### 4. Download the sample PDF

```bash
mkdir -p eval/data
curl -L -o eval/data/docling.pdf https://arxiv.org/pdf/2501.17887
```

### 5. Ingest the PDF

```bash
uv run python -m packages.ingest.ingest eval/data/docling.pdf
# Note the doc_id printed at the end — you'll need it.
# First run downloads BGE-M3 (~1.2GB). Subsequent ingests are fast.
```

### 6. Run everything with one command

```bash
./start.sh                # Chainlit on :8502  (recommended)
./start.sh --streamlit    # Streamlit on :8501 instead
```

`start.sh` checks prereqs, runs `uv sync` if needed, brings up Docker services, starts FastAPI + arq worker in the background (logs in `./logs/`), waits for the API to warm up, then opens the UI in the foreground. Ctrl-C in the UI shuts down the background processes cleanly. Docker services keep running — stop them with `docker compose stop qdrant redis`.

<details><summary>Or run the 3 processes manually</summary>

**Terminal A — FastAPI:**
```bash
uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal B — async ingest worker:**
```bash
uv run arq apps.worker.worker.WorkerSettings
```

**Terminal C — pick ONE UI:**
```bash
uv run chainlit run apps/chainlit_ui/app.py --port 8502 -h
# OR
uv run streamlit run apps/frontend/streamlit_app.py
```
</details>

In either UI, upload a PDF; the upload returns immediately and the UI polls progress as Docling parses, contextual headers generate, BGE-M3 embeds, RAPTOR builds the tree, and Qdrant indexes.

> **Tip**: If you don't want to run the worker, use the sync fallback `POST /upload/sync` (or the CLI in step 5) — useful for testing with the eval script.

### 7. Run the eval

```bash
uv run python -m eval.run_eval --doc-id <doc_id_from_step_5> --dataset docling_arxiv
# Prints PASS/FAIL per query.
```

---

## Sample queries

Full list with expected behavior in [`eval/sample_queries.yaml`](eval/sample_queries.yaml).

**5 valid (in-scope) — must produce cited answers:**

1. *What dataset does Docling's TableFormer use for table structure recognition?*
2. *List the export formats supported by DoclingDocument.*
3. *What is the average conversion time per page reported in the benchmarks?*
4. *Show me the figure that sketches Docling's pipeline.*
5. *Summarize the document's main contributions.*

**3 out-of-scope — must produce the exact refusal string:**

1. *What is Docling's market share compared to Adobe Acrobat in 2025?*
2. *How does Docling compare to Reducto on the RD-TableBench benchmark?*
3. *Translate the entire abstract into Mandarin.*

---

## How grounding actually works

```
user query
   │
   ▼
standalone-question rewrite (Haiku, history-aware)
   │
   ▼
OOS check (cosine vs doc centroid)  ──► refusal
   │
   ▼
hybrid retrieval: dense + sparse via Qdrant Query API
   │   └── prefetch=50, RRF fusion, then bge-reranker-v2-m3 → top-8
   ▼
top-1 rerank score < threshold?  ──► refusal
   │
   ▼
Claude Sonnet 4.5  +  Citations API (custom-content mode)
   │   └── each retrieved chunk → 1 document content block,
   │       citations.enabled = true
   ▼
faithfulness gate (Haiku LLM-judge)
   │   └── score < 0.85 → refusal
   ▼
stream {text, citations} to frontend
```

**Why citations are robust**: the Anthropic Citations API enforces sentence→chunk binding at the API level (not via prompt), so even if the model hallucinates wording, the citation it emits points to a real chunk we sent. The faithfulness gate then catches the case where the citation is wrong.

**Why OOS refusal is robust**: two independent signals — query embedding too far from doc centroid, OR no chunk reranked above the score floor — either triggers the fixed refusal string.

---

## Multilingual (bonus)

BGE-M3 handles cross-lingual retrieval natively: an English query can retrieve Hindi (or Tamil/Bengali/Marathi/…) chunks, and the agent's system prompt forces the answer language to match the question language while citing the original-language source page.

To test:

```bash
curl -L -o eval/data/rbi.pdf "https://rbidocs.rbi.org.in/rdocs/AnnualReport/PDFs/0ANNUALREPORT20222322A548270D6140D998AA20E8207075E4.PDF"
uv run python -m packages.ingest.ingest eval/data/rbi.pdf
uv run python -m eval.run_eval --doc-id <doc_id> --dataset rbi_bilingual
```

---

## What's deferred (Phase 3/4 roadmap)

Phase 1 + Phase 2 are now implemented. Remaining items in the architecture doc:

- **Phase 3**: vision-LLM figure captioning + JinaCLIP-v2 multimodal embeddings, Surya equation→LaTeX, propositions index, NetworkX entity KG, bespoke-minicheck-7B NLI gate, ColPali (GPU-only).
- **Phase 4**: full 100-query eval set, OpenTelemetry latency/cost telemetry, Next.js frontend.

See `~/.claude/plans/ok-so-the-claude-ai-elegant-lightning.md` for the detailed plan.

---

## Repo layout

```
pdf/
├── pyproject.toml
├── docker-compose.yml              # qdrant + redis
├── .env.example
├── packages/
│   ├── core/                       # schema, settings, citations parsing
│   ├── ingest/                     # docling parse, chunker, BGE-M3, qdrant
│   │   ├── contextual.py           # Phase 2: Anthropic Contextual Retrieval
│   │   └── raptor.py               # Phase 2: UMAP+GMM tree builder
│   ├── retrieve/                   # hybrid retrieval, reranker, OOS
│   │   └── query_xform.py          # Phase 2: HyDE + multi-query + step-back + decompose
│   └── agent/                      # prompts, memory, grounding, agent
│       └── verify.py               # Phase 2: two-pass verifier
├── apps/
│   ├── api/                        # FastAPI: /upload (async), /upload/status, /chat (SSE)
│   ├── worker/                     # arq async ingest worker + Redis progress writer
│   ├── frontend/                   # Streamlit demo
│   └── chainlit_ui/                # Chainlit UI with instant/deep chat profiles
├── eval/
│   ├── sample_queries.yaml         # 5 valid + 3 OOS Docling, 2+1 RBI multilingual
│   └── run_eval.py                 # PASS/FAIL harness
└── tests/                          # unit smoke tests
```

---

## Troubleshooting

- **`ANTHROPIC_API_KEY` validation error**: copy `.env.example` to `.env` and set the key.
- **First request is slow**: the FastAPI app warms up BGE-M3 + reranker on boot; expect ~30 s on first start while models download.
- **Qdrant connection refused**: confirm `docker compose ps` shows `qdrant` healthy on port 6333.
- **`No module named 'docling_core'`** or similar Docling import errors: `uv sync` (or `pip install -U docling`); Docling 2.x has fast-moving deps.
- **Streamlit shows "Streaming failed"**: check the API logs — usually means the OOS check fired or Anthropic returned a non-200.
