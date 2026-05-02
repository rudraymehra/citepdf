"""Chat orchestrator with token-by-token streaming.

Instant mode flow:
  rewrite_standalone -> in_scope check -> retrieve -> stream generate
  (text + citations as Anthropic emits them) -> done.

Deep mode flow:
  rewrite_standalone -> in_scope check -> deep_retrieve -> stream generate
  -> verifier (post-stream) -> faithfulness gate (post-stream) ->
  emit `warning` events for any flagged claims -> done.

Why instant mode skips the post-stream gates: the Citations API already
binds every cited sentence to a chunk_id we sent. The LLM-judge gate adds
3-5s of latency for marginal additional safety. Deep mode keeps both gates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import anthropic

from packages.agent.grounding_check import check_faithfulness
from packages.agent.memory import rewrite_standalone
from packages.agent.prompts import SYSTEM_PROMPT
from packages.agent.verify import find_unsupported_claims, run_verifier
from packages.core.citations import chunks_to_documents, citation_from_anthropic, parse_response_blocks
from packages.core.schema import GenerationEvent
from packages.core.settings import get_settings
from packages.retrieve.oos import in_scope_query, retrieval_passes_threshold
from packages.retrieve.query_xform import deep_retrieve
from packages.retrieve.retriever import RetrievedChunk, retrieve

log = logging.getLogger(__name__)


def chat(
    doc_id: str,
    user_message: str,
    history: list[dict[str, str]] | None = None,
    mode: str = "instant",
) -> Iterator[GenerationEvent]:
    import time
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    s = get_settings()
    history = history or []
    refusal = s.refusal_string
    client = anthropic.Anthropic(api_key=s.anthropic_api_key)

    # 1. Rewrite to standalone question
    t = time.perf_counter()
    standalone = rewrite_standalone(user_message, history, client=client)
    timings["1_rewrite"] = time.perf_counter() - t
    log.info("standalone (%.2fs): %s", timings["1_rewrite"], standalone[:200])

    # 2. Cheap OOS check (cosine vs document centroid)
    t = time.perf_counter()
    oos = in_scope_query(doc_id, standalone)
    timings["2_oos_centroid"] = time.perf_counter() - t
    if not oos.in_scope:
        log.info("OOS by centroid (sim=%.3f); refusing", oos.centroid_sim)
        yield GenerationEvent(kind="refusal", text=refusal)
        yield GenerationEvent(kind="done")
        return

    # 3. Retrieve (mode-dependent)
    t = time.perf_counter()
    if mode == "deep":
        chunks = deep_retrieve(doc_id, standalone, client=client)
    else:
        chunks = retrieve(doc_id, standalone)
    timings["3_retrieve"] = time.perf_counter() - t

    if chunks:
        top_scores = [round(c.rerank_score, 3) for c in chunks[:5]]
        log.info(
            "retrieve (%.2fs): n=%d, top-5 rerank scores=%s",
            timings["3_retrieve"],
            len(chunks),
            top_scores,
        )
    if not chunks or not retrieval_passes_threshold(chunks):
        log.info(
            "OOS by retrieval threshold (n=%d, top1=%.3f); refusing",
            len(chunks),
            chunks[0].rerank_score if chunks else 0.0,
        )
        yield GenerationEvent(kind="refusal", text=refusal)
        yield GenerationEvent(kind="done")
        return

    # 4. Stream generate with Citations API
    # Both the system prompt and the document content blocks are marked
    # cache_control=ephemeral. First turn pays full cost; subsequent turns
    # within 5 min that include the same prefix pay 10% on cached portions.
    docs = chunks_to_documents(chunks, cache_documents=True)
    user_content: list[dict[str, Any]] = list(docs) + [
        {"type": "text", "text": standalone}
    ]
    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT.format(refusal=refusal),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    accumulated_text = ""
    citations_collected: list = []
    seen_citation_keys: set[tuple[str, str]] = set()
    t_gen_start = time.perf_counter()
    t_first_token: float | None = None

    try:
        with client.messages.stream(
            model=s.anthropic_generation_model,
            max_tokens=2048,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                if etype != "content_block_delta":
                    continue
                delta = getattr(event, "delta", None)
                if delta is None:
                    continue
                dtype = getattr(delta, "type", "")
                if dtype == "text_delta":
                    text = getattr(delta, "text", "") or ""
                    if text:
                        if t_first_token is None:
                            t_first_token = time.perf_counter() - t_gen_start
                            log.info("4_generate first-token: %.2fs", t_first_token)
                        accumulated_text += text
                        yield GenerationEvent(kind="text", text=text)
                elif dtype == "citations_delta":
                    raw_cit = getattr(delta, "citation", None)
                    if raw_cit is None:
                        continue
                    cit = citation_from_anthropic(raw_cit, chunks)
                    if cit is None:
                        continue
                    key = (cit.chunk_id, (cit.cited_text or "")[:80])
                    if key in seen_citation_keys:
                        continue
                    seen_citation_keys.add(key)
                    citations_collected.append(cit)
                    yield GenerationEvent(kind="citation", citation=cit)
    except Exception as e:
        log.exception("Streaming generation failed")
        yield GenerationEvent(kind="error", error=str(e))
        yield GenerationEvent(kind="done")
        return

    timings["4_generate_total"] = time.perf_counter() - t_gen_start
    log.info(
        "streamed %d chars, %d citations, mode=%s, gen=%.2fs",
        len(accumulated_text),
        len(citations_collected),
        mode,
        timings["4_generate_total"],
    )
    log.info(
        "TIMING: total=%.2fs | rewrite=%.2fs oos=%.2fs retrieve=%.2fs gen=%.2fs (first-token=%.2fs)",
        time.perf_counter() - t_start,
        timings.get("1_rewrite", 0.0),
        timings.get("2_oos_centroid", 0.0),
        timings.get("3_retrieve", 0.0),
        timings["4_generate_total"],
        t_first_token or -1.0,
    )

    # If the model emitted the literal refusal string, surface it
    if accumulated_text.strip() == refusal.strip():
        yield GenerationEvent(kind="done")
        return

    # 5. Deep mode: verifier + CRAG re-retrieval loop (one retry max)
    if mode == "deep" and accumulated_text.strip():
        try:
            verify_result = run_verifier(accumulated_text, chunks, client=client)
        except Exception as e:
            log.warning("Verifier failed: %s", e)
            verify_result = None

        if verify_result and verify_result.unsupported:
            log.info(
                "Verifier flagged %d unsupported claim(s); running CRAG retry",
                len(verify_result.unsupported),
            )
            yield GenerationEvent(
                kind="warning",
                text=(
                    f"⚠️ Verifier flagged {len(verify_result.unsupported)} claim(s) "
                    "as not directly supported. Refining via CRAG re-retrieval…"
                ),
            )

            # CRAG: re-retrieve on each unsupported claim, merge with original chunks,
            # regenerate once, present the refined answer as a follow-up.
            augmented_chunks = _crag_augment_chunks(
                doc_id, chunks, verify_result.unsupported
            )
            refined_text, refined_citations = _generate_non_streaming(
                client, s, system_blocks, augmented_chunks, standalone
            )

            if refined_text.strip() and refined_text.strip() != refusal.strip():
                # Re-verify the refined answer
                try:
                    refined_unsupported = find_unsupported_claims(
                        refined_text, augmented_chunks, client=client
                    )
                except Exception:
                    refined_unsupported = []

                yield GenerationEvent(
                    kind="text",
                    text=f"\n\n---\n**Refined answer (CRAG retry):**\n{refined_text}",
                )
                for cit in refined_citations:
                    yield GenerationEvent(kind="citation", citation=cit)

                if refined_unsupported:
                    yield GenerationEvent(
                        kind="warning",
                        text=(
                            f"⚠️ {len(refined_unsupported)} claim(s) still flagged "
                            "after CRAG retry."
                        ),
                    )
                else:
                    yield GenerationEvent(
                        kind="warning",
                        text="✓ CRAG retry resolved the verifier flags.",
                    )

    # 6. Deep mode only: faithfulness gate (warning, text already streamed)
    if mode == "deep" and accumulated_text.strip():
        try:
            fcheck = check_faithfulness(accumulated_text, chunks, client=client)
            if not fcheck.supported:
                log.warning(
                    "Faithfulness gate failed (score=%.2f, %d unsupported)",
                    fcheck.score,
                    len(fcheck.unsupported_claims),
                )
                yield GenerationEvent(
                    kind="warning",
                    text=(
                        f"⚠️ Faithfulness check: {fcheck.score:.0%} "
                        f"(below {s.faithfulness_threshold:.0%} threshold)."
                    ),
                )
        except Exception as e:
            log.warning("Faithfulness check failed: %s", e)

    yield GenerationEvent(kind="done")


def _crag_augment_chunks(
    doc_id: str,
    original: list[RetrievedChunk],
    unsupported_claims: list[str],
    per_claim_top_k: int = 4,
    max_claims: int = 3,
) -> list[RetrievedChunk]:
    """Re-retrieve on each unsupported claim and merge with the original chunks."""
    seen: set[str] = {c.chunk_id for c in original}
    augmented = list(original)
    for claim in unsupported_claims[:max_claims]:
        if not claim or len(claim) < 5:
            continue
        try:
            extra = retrieve(doc_id, claim, top_k=per_claim_top_k)
        except Exception as e:
            log.warning("CRAG re-retrieval failed for claim: %s", e)
            continue
        for c in extra:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            augmented.append(c)
    log.info(
        "CRAG augmented chunk pool: %d -> %d (added %d)",
        len(original),
        len(augmented),
        len(augmented) - len(original),
    )
    return augmented


def _generate_non_streaming(
    client: anthropic.Anthropic,
    s,
    system_blocks: list[dict[str, Any]],
    chunks: list[RetrievedChunk],
    standalone: str,
) -> tuple[str, list]:
    """One-shot generate (used by CRAG retry, no streaming)."""
    docs = chunks_to_documents(chunks, cache_documents=False)
    user_content: list[dict[str, Any]] = list(docs) + [
        {"type": "text", "text": standalone}
    ]
    try:
        resp = client.messages.create(
            model=s.anthropic_generation_model,
            max_tokens=2048,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as e:
        log.exception("CRAG retry generation failed")
        return f"[CRAG retry failed: {e}]", []
    return parse_response_blocks(resp.content, chunks)
