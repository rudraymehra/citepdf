"""Anthropic Citations API helpers — custom-content mode.

Each retrieved Chunk becomes one document content block sent to Claude. The
`document_index` returned in citations maps back to our chunk metadata
(page, section_anchor, image_ref, ...) so we can render citations as
[p. 47, §3.2 Methods] in the UI.

References:
- https://platform.claude.com/docs/en/build-with-claude/citations
- https://docs.claude.com/en/api/messages-examples#citations
"""

from __future__ import annotations

import logging
from typing import Any

from packages.core.schema import Citation
from packages.retrieve.retriever import RetrievedChunk

log = logging.getLogger(__name__)


def chunks_to_documents(
    chunks: list[RetrievedChunk], cache_documents: bool = True
) -> list[dict[str, Any]]:
    """Build the list of Anthropic document content blocks for the API call.

    Custom-content mode: source.type=content with one text block per chunk.
    Title is a short human-readable label that Anthropic echoes in citations.

    When `cache_documents` is True (default), the LAST document gets a
    `cache_control: ephemeral` marker — this caches the entire prefix
    (system prompt + all documents up through the last one) at Anthropic's
    side. Subsequent turns within 5 minutes pay 10% on the cached portion."""
    docs: list[dict[str, Any]] = []
    n = len(chunks)
    for i, c in enumerate(chunks):
        page = c.page_start
        if c.page_end and c.page_end != page:
            page_label = f"pp.{c.page_start}-{c.page_end}"
        else:
            page_label = f"p.{page}"
        section = c.section_label or "?"
        title = f"chunk {i} | {page_label} | {section}"[:200]

        block: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "content",
                "content": [{"type": "text", "text": c.text}],
            },
            "title": title,
            "context": _context_for(c),
            "citations": {"enabled": True},
        }
        # Cache the last document — extends caching back through all prior
        # documents and the system prompt prefix. Anthropic supports up to
        # 4 cache breakpoints; we use 1 for the chat path and 1 for the
        # system prompt (set by the caller).
        if cache_documents and i == n - 1:
            block["cache_control"] = {"type": "ephemeral"}
        docs.append(block)
    return docs


def _context_for(c: RetrievedChunk) -> str:
    block_types = c.payload.get("block_types") or []
    pieces = []
    if block_types:
        pieces.append("type=" + ",".join(set(block_types)))
    if c.payload.get("language"):
        pieces.append("lang=" + c.payload["language"])
    return " | ".join(pieces)


def citation_from_anthropic(
    raw: Any, chunks: list[RetrievedChunk]
) -> Citation | None:
    """Convert one Anthropic-returned citation dict/object to our Citation model."""
    # The SDK exposes citations as objects with attribute access; we tolerate
    # both attribute and dict access for forward-compat.
    def get(key: str, default: Any = None) -> Any:
        if isinstance(raw, dict):
            return raw.get(key, default)
        return getattr(raw, key, default)

    doc_idx = get("document_index")
    if doc_idx is None or doc_idx >= len(chunks):
        return None
    chunk = chunks[doc_idx]
    cited_text = get("cited_text", "") or ""
    section_label = chunk.section_label
    raw_bboxes = chunk.payload.get("block_bboxes") or []
    bboxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for entry in raw_bboxes:
        try:
            page = int(entry[0])
            box = entry[1]
            bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            bboxes.append((page, bbox))
        except (TypeError, ValueError, IndexError):
            continue
    return Citation(
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_anchor=chunk.section_anchor,
        section_label=section_label,
        cited_text=cited_text,
        block_types=list(chunk.payload.get("block_types") or []),
        image_ref=chunk.payload.get("image_ref"),
        bboxes=bboxes,
    )


def parse_response_blocks(
    response_content: list[Any], chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Walk a non-streaming Anthropic response and gather (text, citations).

    Each text block may carry a `citations` list. We flatten everything into
    a single text + a list of Citation objects."""
    parts: list[str] = []
    cites: list[Citation] = []
    for block in response_content:
        if getattr(block, "type", None) == "text" or (
            isinstance(block, dict) and block.get("type") == "text"
        ):
            text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text", "")
            parts.append(text or "")
            raw_cites = (
                getattr(block, "citations", None)
                if not isinstance(block, dict)
                else block.get("citations")
            )
            if raw_cites:
                for rc in raw_cites:
                    parsed = citation_from_anthropic(rc, chunks)
                    if parsed is not None:
                        cites.append(parsed)
    return "".join(parts), cites
