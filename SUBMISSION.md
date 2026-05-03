# Task 3 — Submission

> **Live demo**: `https://huggingface.co/spaces/<your-username>/pdf-rag`

## What this is

A PDF-constrained conversational RAG agent. You upload a PDF; it chats about that PDF with **strict grounding**, **citations on every factual claim**, and refuses out-of-scope questions with the exact line:

> _I cannot answer this from the provided document._

## How to test it (5 minutes)

1. Open the live demo URL above. You'll land in Chainlit.
2. **Pick a chat profile** at the top: `instant` (~3 s, baseline) or `deep` (~30 s, strongest grounding — HyDE + multi-query + step-back + decomposition + two-pass verifier). Deep mode is the one to stress-test.
3. **Upload a PDF.** A progress bar shows the pipeline phases: parsing → language detect → chunking → contextual headers → embedding → RAPTOR tree → indexing.
4. **Ask the test queries below.** All 8 are in `eval/sample_queries.yaml`.

### Suggested PDF for the demo

The Docling arXiv paper (`https://arxiv.org/pdf/2501.17887`). It exercises every modality: multi-column text, tables, equations, figures, footnotes. Or upload your own — any PDF works.

### 5 in-scope queries (must produce cited answers)

1. _What dataset does Docling's TableFormer use for table structure recognition?_
2. _List the export formats supported by DoclingDocument._
3. _What is the average conversion time per page reported in the benchmarks?_
4. _Show me the figure that sketches Docling's pipeline._
5. _Summarize the document's main contributions._  ← stresses RAPTOR; flat retrieval would struggle here

### 3 out-of-scope queries (must produce the exact refusal string)

1. _What is Docling's market share compared to Adobe Acrobat in 2025?_  ← not in document
2. _How does Docling compare to Reducto on the RD-TableBench benchmark?_  ← paper doesn't benchmark Reducto
3. _Translate the entire abstract into Mandarin._  ← transformation outside grounding

### What to look for in answers

- Every factual sentence has a citation footnote in the side panel (chunk text + `[p. X, §Y]`)
- Numbers are quoted **verbatim** from tables, not paraphrased
- Refusals are **exact** — same string every time, no clever wording around it
- Cross-lingual works: ask a question in Hindi about an English PDF — the answer comes in Hindi but cites the original English page

## Architecture (1-page summary)

| Layer | Choice | Why |
|---|---|---|
| Parse | **Docling 2.x** | Best open-source layout AI; multi-column reading order; TableFormer for tables |
| Chunk | **Layout-aware flat** + atomic tables/figures + intra-section overlap | Tables/figures must never split; section boundaries respected for retrieval precision |
| Context | **Anthropic Contextual Retrieval headers** with **prompt caching** of full doc body | Up to 67% retrieval-error reduction (Anthropic, Sep 2024); cache amortizes per-chunk cost |
| Tree | **RAPTOR** (UMAP+GMM clustering, L1 cluster + L2 section + L3 doc summaries by Claude Opus 4.7) | Holistic queries ("summarize main contributions") need cross-chunk aggregation that flat retrieval can't do |
| Embed | **BGE-M3** dense + sparse in one model | 1024-dim dense + learned sparse, 100+ languages incl. Hindi/Tamil/Telugu/Bengali/Marathi/Gujarati/Kannada/Malayalam/Punjabi |
| Index | **Qdrant 1.12** with native Query API hybrid (dense + sparse, RRF fusion server-side) | Single round-trip hybrid; multi-vector ready for Phase 3 ColPali |
| Rerank | **bge-reranker-v2-m3** cross-encoder | Multilingual; 50 → 8 candidates with strong MRR lift |
| Generate | **Claude Opus 4.7** + **Anthropic Citations API** (custom-content mode) | Sentence-level chunk binding enforced at the API level, not via prompt |
| Verify (deep) | Two-pass: **Opus** verifier drops unsupported claims, then **Sonnet 4.6 LLM-judge** faithfulness gate | Below threshold → fixed refusal string |
| OOS detect | (a) cosine vs document centroid, (b) top-1 rerank score floor | Two independent signals before any generation |
| Async | **arq + Redis** worker + per-doc Redis progress hash | Frontend polls `/upload/status/{doc_id}` for phase-by-phase progress |
| UI | **Chainlit** (with `instant`/`deep` chat profiles) + Streamlit fallback | "Skipper-style" profile picker maps cleanly to mode toggle |

## Key design decisions worth defending

1. **Tree-aware retrieval with citation back-traces** — every L1/L2/L3 summary node carries `child_ids` pointing to its underlying L0 leaves. The retriever expands summary hits to leaves before sending to Claude, so the answer always cites a citable page, never a paraphrased summary.

2. **Anthropic Citations API in custom-content mode** — each retrieved chunk becomes one `document` content block. The model can't fabricate a citation: any returned `document_index` maps deterministically back to a chunk we sent. Hallucinated citations are impossible by construction.

3. **Two independent OOS signals** — query embedding cosine vs document centroid, AND retrieval rerank score floor. Either trip → fixed refusal string. This is why the agent doesn't get fooled by adversarial-but-plausible queries.

4. **Prompt-caching the full document body** during contextual-header generation — first call pays full corpus tokens, every subsequent chunk pays 10%. For a 200-page PDF, this turns a $5 ingest into ~$0.50.

5. **Self-hosted ML stack** (BGE-M3 + bge-reranker-v2-m3 + Surya) so the only paid dependency is Anthropic. If you swap in your own embeddings/reranker, no other code changes.

## Evaluation

```bash
uv run python -m eval.run_eval --doc-id <id> --dataset docling_arxiv
```

Runs the full 8-query test (5 valid + 3 OOS), then computes RAGAS-style metrics using Claude as the LLM judge:
- **Faithfulness** — fraction of answer claims supported by retrieved chunks
- **Answer Relevancy** — does the answer address the question?
- **Citation Accuracy** — fraction of citations whose page falls in the expected range
- **Refusal Precision / Recall** — how precise/complete is the OOS refusal behavior

Latest run results are written to [`eval/results.md`](eval/results.md) — a markdown table per query plus aggregate scores. Reviewers can re-run with `uv run python -m eval.run_eval --doc-id <id>` and verify the numbers themselves.

The harness also covers a multilingual slice (`--dataset rbi_bilingual`) for the bonus marks.

## What's deferred (Phase 3)

vision-LLM figure captioning + JinaCLIP-v2 multimodal embeddings, Surya equation→LaTeX, propositions index, NetworkX entity KG, bespoke-minicheck NLI gate, ColPali. The Phase 1 + 2 codebase is structured so each is a clean addition.

---

Questions or anything broken in the demo? **<your-email-or-handle>**
