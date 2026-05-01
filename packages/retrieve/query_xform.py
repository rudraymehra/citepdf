"""Query transformations for deep mode.

  - HyDE: Haiku writes a hypothetical answer; embed THAT instead of the raw query.
  - Multi-query: generate N paraphrases; union their retrievals via RRF.
  - Step-back: emit a more abstract version (helps with overly specific queries).
  - Decompose: for genuine multi-hop questions, split into sub-queries.

deep_retrieve() applies all transformations, runs each query through the
hybrid retriever, fuses by RRF, then reranks. Returns top-k L0 leaves.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import anthropic

from packages.core.settings import get_settings
from packages.retrieve.rerank import get_reranker
from packages.retrieve.retriever import RetrievedChunk, retrieve

log = logging.getLogger(__name__)

RRF_K = 60


HYDE_PROMPT = """Write a brief hypothetical answer paragraph (3-5 sentences) to the user's question, as if you were quoting from a real document. Use plausible technical terms and concrete details. Do NOT add disclaimers or preambles. Output the paragraph only.

Question: {query}

Hypothetical answer:"""


MULTIQUERY_PROMPT = """Generate {n} alternative paraphrases of this query for retrieval. Each paraphrase should preserve the user's intent but use different vocabulary or framing.

Output only the paraphrases, one per line, no numbering, no preamble.

Original query: {query}

Paraphrases:"""


STEPBACK_PROMPT = """Write a more abstract, higher-level version of this query that asks about the broader topic. The step-back query should be useful for retrieving general context. Output only the step-back query, no preamble.

Original query: {query}

Step-back query:"""


DECOMPOSE_CLASSIFIER_PROMPT = """Does answering this question require combining facts from multiple separate sections or paragraphs of a document? Reply with EXACTLY one of: YES or NO. No explanation.

Query: {query}

Answer:"""


DECOMPOSE_PROMPT = """Decompose this multi-hop question into 2-4 atomic sub-questions, each answerable from a single section. Output one sub-question per line, no numbering, no preamble.

Multi-hop query: {query}

Sub-questions:"""


def hyde(query: str, client: anthropic.Anthropic) -> str:
    s = get_settings()
    try:
        r = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=200,
            temperature=0.3,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        )
        return "".join(b.text for b in r.content if hasattr(b, "text")).strip() or query
    except Exception as e:
        log.warning("HyDE failed: %s", e)
        return query


def multi_query(query: str, n: int, client: anthropic.Anthropic) -> list[str]:
    s = get_settings()
    try:
        r = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=300,
            temperature=0.5,
            messages=[
                {"role": "user", "content": MULTIQUERY_PROMPT.format(query=query, n=n)}
            ],
        )
        text = "".join(b.text for b in r.content if hasattr(b, "text")).strip()
        lines = [ln.strip(" -•\t") for ln in text.split("\n") if ln.strip()]
        return lines[:n]
    except Exception as e:
        log.warning("multi_query failed: %s", e)
        return []


def step_back(query: str, client: anthropic.Anthropic) -> str:
    s = get_settings()
    try:
        r = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=120,
            temperature=0.0,
            messages=[{"role": "user", "content": STEPBACK_PROMPT.format(query=query)}],
        )
        return "".join(b.text for b in r.content if hasattr(b, "text")).strip()
    except Exception as e:
        log.warning("step_back failed: %s", e)
        return ""


def decompose_if_multihop(query: str, client: anthropic.Anthropic) -> list[str]:
    s = get_settings()
    try:
        r = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=10,
            temperature=0.0,
            messages=[
                {"role": "user", "content": DECOMPOSE_CLASSIFIER_PROMPT.format(query=query)}
            ],
        )
        verdict = "".join(b.text for b in r.content if hasattr(b, "text")).strip().upper()
        if not verdict.startswith("YES"):
            return []
        r2 = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=400,
            temperature=0.0,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}],
        )
        text = "".join(b.text for b in r2.content if hasattr(b, "text")).strip()
        subs = [ln.strip(" -•\t?") for ln in text.split("\n") if ln.strip()]
        return [s + "?" if not s.endswith("?") else s for s in subs[:4]]
    except Exception as e:
        log.warning("decompose failed: %s", e)
        return []


def deep_retrieve(
    doc_id: str,
    query: str,
    top_k: int | None = None,
    client: anthropic.Anthropic | None = None,
) -> list[RetrievedChunk]:
    """Run all transforms, union retrievals via RRF, rerank, return top-k leaves."""
    s = get_settings()
    top_k = top_k or s.topk_final
    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)

    # Generate transformed queries in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_hyde = pool.submit(hyde, query, client)
        f_multi = pool.submit(multi_query, query, s.deep_mode_multiquery_n, client)
        f_step = pool.submit(step_back, query, client)
        f_decomp = pool.submit(decompose_if_multihop, query, client)
        hyde_text = f_hyde.result()
        paraphrases = f_multi.result()
        sb_text = f_step.result()
        sub_qs = f_decomp.result()

    queries: list[str] = [query]
    if hyde_text and hyde_text != query:
        queries.append(hyde_text)
    queries.extend(paraphrases)
    if sb_text:
        queries.append(sb_text)
    queries.extend(sub_qs)
    # Cap fanout
    queries = queries[:8]
    log.info("deep_retrieve: %d transformed queries", len(queries))

    # Run hybrid retrieval for each query in parallel.
    def run_one(q: str) -> list[RetrievedChunk]:
        return retrieve(doc_id, q, top_k=s.topk_prefetch, prefetch_k=s.topk_prefetch)

    with ThreadPoolExecutor(max_workers=4) as pool:
        result_lists = list(pool.map(run_one, queries))

    # RRF fuse the union
    rrf: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, c in enumerate(results):
            rrf[c.chunk_id] += 1.0 / (RRF_K + rank + 1)
            if c.chunk_id not in chunk_by_id:
                chunk_by_id[c.chunk_id] = c

    fused = sorted(chunk_by_id.values(), key=lambda c: rrf[c.chunk_id], reverse=True)
    if not fused:
        return []

    # Final cross-encoder rerank against the original user query (not the transforms)
    reranker = get_reranker()
    docs = [c.text for c in fused[: s.topk_prefetch]]
    rerank_scores = reranker.score(query, docs)
    for c, r in zip(fused[: s.topk_prefetch], rerank_scores):
        c.rerank_score = float(r)

    fused[: s.topk_prefetch].sort(key=lambda c: c.rerank_score, reverse=True)
    return fused[:top_k]
