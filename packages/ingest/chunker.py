"""Layout-aware flat chunker (Phase 1).

Rules (per the architecture doc §3.4):
- Tables and figures are atomic. Each becomes its own chunk regardless of size.
- Code blocks are atomic up to MAX_CHARS.
- Text is packed into ~500 token chunks (≈2000 chars), 50 token (≈200 char) overlap,
  intra-section only — never across section boundaries.
- Headings are merged into the start of the next text chunk in the same section
  to preserve structural context for retrieval.
"""

from __future__ import annotations

import logging

from packages.core.schema import BlockNode, Chunk
from .lang import dominant_language

log = logging.getLogger(__name__)

# Heuristic: ~4 chars/token for Latin scripts. Target ~500 tokens, max ~800,
# overlap ~50 tokens. Tables/figures are exempt.
TARGET_CHARS = 2000
MAX_CHARS = 3200
OVERLAP_CHARS = 200


def chunk_blocks(blocks: list[BlockNode], doc_id: str) -> list[Chunk]:
    """Group BlockNodes into Chunks following the layout-aware rules."""
    chunks: list[Chunk] = []
    current_section_anchor: str | None = None
    pending_heading: BlockNode | None = None
    text_buffer: list[BlockNode] = []

    def flush_text() -> None:
        nonlocal text_buffer, pending_heading
        if not text_buffer:
            return
        prefix = [pending_heading] if pending_heading is not None else []
        chunks.extend(_pack_with_overlap(prefix + text_buffer, doc_id))
        text_buffer = []
        pending_heading = None

    for block in blocks:
        if block.section_anchor != current_section_anchor:
            flush_text()
            current_section_anchor = block.section_anchor

        if block.block_type == "heading":
            flush_text()
            pending_heading = block
            continue

        if block.block_type in ("table", "figure", "code"):
            flush_text()
            chunks.append(_atomic_chunk_from_block(block, doc_id))
            continue

        # text / footnote / caption / equation
        text_buffer.append(block)

    flush_text()
    log.info("Chunker emitted %d chunks from %d blocks", len(chunks), len(blocks))
    return chunks


def _pack_with_overlap(blocks: list[BlockNode], doc_id: str) -> list[Chunk]:
    """Greedy packing with intra-pack overlap from prior pack's tail."""
    out: list[Chunk] = []
    pack: list[BlockNode] = []
    pack_chars = 0
    pending_overlap = ""

    def flush() -> None:
        nonlocal pack, pack_chars, pending_overlap
        if not pack:
            return
        parts = [pending_overlap] if pending_overlap else []
        parts.extend(b.text for b in pack if b.text)
        text = "\n\n".join(p for p in parts if p).strip()
        if text:
            out.append(_chunk_from_blocks(pack, text, doc_id))
        pending_overlap = _tail(pack, OVERLAP_CHARS)
        pack = []
        pack_chars = 0

    for b in blocks:
        b_chars = len(b.text)
        if pack and pack_chars + b_chars > MAX_CHARS:
            flush()
        pack.append(b)
        pack_chars += b_chars + 2

    flush()
    return out


def _tail(blocks: list[BlockNode], n_chars: int) -> str:
    full = "\n\n".join(b.text for b in blocks if b.text)
    if len(full) <= n_chars:
        return ""
    tail = full[-n_chars:]
    for sep in [". ", "\n", " "]:
        idx = tail.find(sep)
        if 0 <= idx < n_chars // 2:
            return tail[idx + len(sep):]
    return tail


def _chunk_from_blocks(blocks: list[BlockNode], text: str, doc_id: str) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        level=0,
        text=text,
        source_block_ids=[b.block_id for b in blocks],
        pages=sorted({b.page for b in blocks}),
        section_paths=_unique_paths([b.section_path for b in blocks]),
        section_anchor=blocks[0].section_anchor,
        block_types=[b.block_type for b in blocks],
        language=dominant_language([b.language for b in blocks]),
        block_bboxes=[(b.page, tuple(b.bbox)) for b in blocks if b.bbox],
    )


def _atomic_chunk_from_block(block: BlockNode, doc_id: str) -> Chunk:
    """Tables, figures, code blocks: each is its own atomic chunk."""
    text = block.text
    if block.block_type == "table" and block.html_table:
        text = f"{block.text}\n\n[table HTML preserved in payload]"
    if block.block_type == "figure" and block.image_caption:
        text = f"Figure caption: {block.image_caption}"

    return Chunk(
        doc_id=doc_id,
        level=0,
        text=text or "[empty]",
        source_block_ids=[block.block_id],
        pages=[block.page],
        section_paths=[block.section_path] if block.section_path else [],
        section_anchor=block.section_anchor,
        block_types=[block.block_type],
        language=block.language,
        html_table=block.html_table,
        latex=block.latex,
        image_ref=block.image_ref,
        image_caption=block.image_caption,
        block_bboxes=[(block.page, tuple(block.bbox))] if block.bbox else [],
    )


def _unique_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for p in paths:
        tup = tuple(p)
        if tup and tup not in seen:
            seen.add(tup)
            out.append(p)
    return out
