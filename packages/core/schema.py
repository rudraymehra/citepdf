"""Shared pydantic models.

BlockNode: the unified parsed-PDF block (post-Docling, pre-chunking).
Chunk: an index unit (one or more BlockNodes joined under chunking rules).
Citation: a parsed Anthropic Citations API citation, re-attached to our metadata.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

BlockType = Literal[
    "text",
    "heading",
    "table",
    "figure",
    "equation",
    "code",
    "footnote",
    "caption",
    "form_field",
    "annotation",
]


def _new_id() -> str:
    return str(uuid.uuid4())


class BlockNode(BaseModel):
    """A single layout-aware block extracted from a PDF page."""

    doc_id: str
    block_id: str = Field(default_factory=_new_id)
    page: int
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    block_type: BlockType
    text: str = ""
    latex: str | None = None
    html_table: str | None = None
    image_ref: str | None = None
    image_caption: str | None = None
    language: str = "en"
    section_path: list[str] = Field(default_factory=list)
    section_anchor: str | None = None
    char_offsets: tuple[int, int] = (0, 0)
    reading_order: int = 0
    confidence: float = 1.0
    provenance: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An indexable unit. In Phase 1 every chunk is a layout-aware leaf
    (level=0). Phase 2 adds RAPTOR internal nodes (level>=1) where
    `source_block_ids` traces back to the L0 blocks that informed a summary."""

    doc_id: str
    chunk_id: str = Field(default_factory=_new_id)
    level: int = 0  # 0=leaf, 1=cluster, 2=section, 3=doc (Phase 2+)
    text: str
    contextual_header: str = ""  # Phase 2 — Anthropic Contextual Retrieval prefix

    # Provenance — always populated, even at higher levels
    source_block_ids: list[str] = Field(default_factory=list)
    # For non-leaf nodes (level >= 1), the chunk_ids of the underlying L0
    # leaves this summary was built from. Used at retrieval time to expand a
    # summary hit into its citable leaves.
    child_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    section_paths: list[list[str]] = Field(default_factory=list)
    section_anchor: str | None = None
    block_types: list[str] = Field(default_factory=list)
    language: str = "en"

    # Modality payloads carried on leaves
    html_table: str | None = None
    latex: str | None = None
    image_ref: str | None = None
    image_caption: str | None = None

    # Per-block bounding boxes (page, [l, t, r, b] in PDF points) — used by
    # the PDF-viewer-with-highlights citation renderer. One entry per source
    # BlockNode; multiple if a chunk spans pages or includes multiple blocks.
    block_bboxes: list[tuple[int, tuple[float, float, float, float]]] = Field(
        default_factory=list
    )

    @property
    def page_start(self) -> int:
        return min(self.pages) if self.pages else 0

    @property
    def page_end(self) -> int:
        return max(self.pages) if self.pages else 0

    @property
    def display_section(self) -> str:
        """Human-friendly section label for citations like '§3.2 Methods'."""
        if not self.section_paths:
            return ""
        # All paths in a leaf are typically the same; pick first
        return " > ".join(self.section_paths[0])


class Citation(BaseModel):
    """A normalized citation attached to a generated sentence/span.

    Anthropic's Citations API returns `cited_text` + `start_char_index` +
    `end_char_index` + `document_index`. We map `document_index` back to the
    chunk's metadata (page, section_anchor, block_type, bboxes)."""

    chunk_id: str
    page_start: int
    page_end: int
    section_anchor: str | None = None
    section_label: str = ""  # display string e.g. "§3.2 Methods"
    cited_text: str = ""
    block_types: list[str] = Field(default_factory=list)
    image_ref: str | None = None
    # Page + bbox in PDF-point coords for the cited region(s). Used by the
    # Chainlit highlight viewer to render the page with yellow-box overlays.
    bboxes: list[tuple[int, tuple[float, float, float, float]]] = Field(
        default_factory=list
    )

    def render(self) -> str:
        """One-line bracketed render for inline display."""
        if self.page_start == self.page_end:
            page = f"p. {self.page_start}"
        else:
            page = f"pp. {self.page_start}–{self.page_end}"
        if self.section_label:
            return f"[{page}, {self.section_label}]"
        return f"[{page}]"


class GenerationEvent(BaseModel):
    """Streaming SSE event from /chat.

    Kinds:
      - text: a delta of generated text (multiple events per turn during streaming)
      - citation: an Anthropic Citations API citation, resolved to our chunk metadata
      - warning: soft flag (e.g., faithfulness gate triggered after stream completed)
      - refusal: hard refusal with the fixed refusal string
      - error: terminal error, stream stops
      - done: end of turn
    """

    kind: Literal["text", "citation", "done", "refusal", "warning", "error"]
    text: str | None = None
    citation: Citation | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    doc_id: str
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: Literal["instant", "deep"] = "instant"


class UploadResponse(BaseModel):
    doc_id: str
    n_blocks: int
    n_chunks: int
    pages: int
    languages: list[str]
