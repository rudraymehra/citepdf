"""Anthropic Contextual Retrieval headers (Anthropic, Sep 2024).

For every leaf chunk, generate a 50–100 token "this chunk is from §X about Y"
blurb using Claude Haiku 4.5, then prepend it to the chunk text BEFORE
embedding. Anthropic reports up to 67% retrieval-error reduction.

Cost is amortized via Anthropic prompt caching:
- The full document corpus is sent once as a cached system block.
- Every subsequent chunk call pays only 10% of the corpus tokens (cache read).

We run chunk requests in parallel via a small thread pool. For very large
PDFs (>150K cached tokens), we fall back to per-chunk-only context (no doc
prefix) — quality drops slightly but the system stays usable.

Reference:
- https://www.anthropic.com/news/contextual-retrieval
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from packages.core.schema import BlockNode, Chunk
from packages.core.settings import get_settings

log = logging.getLogger(__name__)

# Roughly ~150K tokens fits in Claude's 200K context with headroom for the
# chunk + response. 4 chars per token heuristic.
MAX_CACHE_CHARS = 600_000
HEADER_MAX_TOKENS = 150
THREAD_POOL_WORKERS = 8

CONTEXT_SYSTEM = """You write short context headers for retrieval chunks of a long document.

For each chunk you receive, write a single English paragraph of 50–100 tokens that:
- Locates the chunk in the document (which section, what surrounds it)
- States what the chunk is specifically about, in plain language
- Avoids quoting the chunk verbatim and avoids generic filler

Output ONLY the header paragraph, no preamble, no quotes, no markdown."""

CONTEXT_USER_TEMPLATE = """Here is the chunk we need a context header for:

<chunk>
{chunk_text}
</chunk>

Write a 50–100 token context header paragraph for this chunk based on the document above."""


def build_doc_corpus(blocks: list[BlockNode], max_chars: int = MAX_CACHE_CHARS) -> str | None:
    """Concatenate all block text with light per-block markers.

    Returns the corpus or None if the document is too large to cache as a
    whole (caller should fall back to no-corpus contextualization)."""
    parts: list[str] = []
    total = 0
    for b in blocks:
        text = (b.text or "").strip()
        if not text:
            continue
        marker = f"[p.{b.page} §{b.section_anchor or '?'} {b.block_type}]"
        piece = f"{marker} {text}"
        if total + len(piece) + 2 > max_chars:
            log.warning(
                "Document too large for full-doc cached prefix (>%d chars). "
                "Falling back to no-prefix contextualization.",
                max_chars,
            )
            return None
        parts.append(piece)
        total += len(piece) + 2
    if not parts:
        return None
    return "\n\n".join(parts)


def add_contextual_headers(
    chunks: list[Chunk],
    blocks: list[BlockNode],
    client: anthropic.Anthropic | None = None,
) -> list[Chunk]:
    """Mutate `chunks` in-place by setting `contextual_header`. Returns the same list."""
    s = get_settings()
    if not chunks:
        return chunks

    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)
    corpus = build_doc_corpus(blocks)

    # Build the (possibly cached) system blocks once
    if corpus is not None:
        system_blocks = [
            {"type": "text", "text": CONTEXT_SYSTEM},
            {
                "type": "text",
                "text": f"<document>\n{corpus}\n</document>",
                "cache_control": {"type": "ephemeral"},
            },
        ]
    else:
        system_blocks = [{"type": "text", "text": CONTEXT_SYSTEM}]

    def gen_header(idx: int, chunk: Chunk) -> tuple[int, str]:
        try:
            resp = client.messages.create(
                model=s.anthropic_judge_model,
                max_tokens=HEADER_MAX_TOKENS,
                temperature=0.0,
                system=system_blocks,
                messages=[
                    {
                        "role": "user",
                        "content": CONTEXT_USER_TEMPLATE.format(chunk_text=chunk.text[:6000]),
                    }
                ],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            return idx, text
        except Exception as e:
            log.warning("contextual header failed for chunk %d: %s", idx, e)
            return idx, ""

    log.info(
        "Generating contextual headers for %d chunks (%s prefix)…",
        len(chunks),
        "cached doc" if corpus else "no prefix",
    )
    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as pool:
        futures = [pool.submit(gen_header, i, c) for i, c in enumerate(chunks)]
        for fut in as_completed(futures):
            i, header = fut.result()
            chunks[i].contextual_header = header

    n_filled = sum(1 for c in chunks if c.contextual_header)
    log.info("Contextual headers: %d/%d filled", n_filled, len(chunks))
    return chunks
