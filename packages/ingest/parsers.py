"""Docling-based PDF parser. Emits a flat list[BlockNode] in reading order.

Phase 1 scope: text, headings, tables (HTML + flat text), figure captions.
Phase 3 will add: equation -> LaTeX (Surya), figure image bytes + multimodal embed.
"""

from __future__ import annotations

import logging
from typing import Any

from packages.core.schema import BlockNode

log = logging.getLogger(__name__)


def parse_pdf(pdf_path: str, doc_id: str) -> list[BlockNode]:
    """Parse a PDF with Docling, return BlockNodes in reading order."""
    from docling.document_converter import DocumentConverter

    log.info("Docling: converting %s", pdf_path)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return _doc_to_blocks(result.document, doc_id)


def _doc_to_blocks(doc: Any, doc_id: str) -> list[BlockNode]:
    blocks: list[BlockNode] = []
    section_stack: list[tuple[int, str]] = []  # (level, label)
    reading_order = 0

    # Docling's iterate_items yields (item, level) tuples in reading order.
    for item, level in doc.iterate_items():
        item_type = type(item).__name__
        page, bbox = _get_prov(item)

        if item_type in ("SectionHeaderItem", "TitleItem"):
            text = _safe_text(item)
            if not text.strip():
                reading_order += 1
                continue
            section_stack = [(lv, lbl) for lv, lbl in section_stack if lv < level]
            section_stack.append((level, text))
            blocks.append(
                BlockNode(
                    doc_id=doc_id,
                    page=page,
                    bbox=bbox,
                    block_type="heading",
                    text=text,
                    section_path=[lbl for _, lbl in section_stack],
                    section_anchor=_anchor_for(section_stack),
                    reading_order=reading_order,
                    provenance={"parser": "docling-2.x", "item_type": item_type},
                )
            )

        elif item_type in ("TextItem", "ListItem", "ParagraphItem"):
            text = _safe_text(item)
            if not text.strip():
                reading_order += 1
                continue
            label = getattr(item, "label", "")
            block_type = "footnote" if "footnote" in str(label).lower() else (
                "caption" if "caption" in str(label).lower() else "text"
            )
            blocks.append(
                BlockNode(
                    doc_id=doc_id,
                    page=page,
                    bbox=bbox,
                    block_type=block_type,
                    text=text,
                    section_path=[lbl for _, lbl in section_stack],
                    section_anchor=_anchor_for(section_stack),
                    reading_order=reading_order,
                    provenance={"parser": "docling-2.x", "item_type": item_type},
                )
            )

        elif item_type == "TableItem":
            html = _table_to_html(item)
            text = _table_to_text(item)
            blocks.append(
                BlockNode(
                    doc_id=doc_id,
                    page=page,
                    bbox=bbox,
                    block_type="table",
                    text=text,
                    html_table=html,
                    section_path=[lbl for _, lbl in section_stack],
                    section_anchor=_anchor_for(section_stack),
                    reading_order=reading_order,
                    provenance={"parser": "docling-2.x", "item_type": item_type},
                )
            )

        elif item_type == "PictureItem":
            caption = _safe_caption(item)
            text = caption or "[figure]"
            blocks.append(
                BlockNode(
                    doc_id=doc_id,
                    page=page,
                    bbox=bbox,
                    block_type="figure",
                    text=text,
                    image_caption=caption,
                    section_path=[lbl for _, lbl in section_stack],
                    section_anchor=_anchor_for(section_stack),
                    reading_order=reading_order,
                    provenance={"parser": "docling-2.x", "item_type": item_type},
                )
            )

        elif item_type == "CodeItem":
            text = _safe_text(item)
            if not text.strip():
                reading_order += 1
                continue
            blocks.append(
                BlockNode(
                    doc_id=doc_id,
                    page=page,
                    bbox=bbox,
                    block_type="code",
                    text=text,
                    section_path=[lbl for _, lbl in section_stack],
                    section_anchor=_anchor_for(section_stack),
                    reading_order=reading_order,
                    provenance={"parser": "docling-2.x", "item_type": item_type},
                )
            )
        # Other items (form fields, annotations, key-value, etc.) are skipped in Phase 1.

        reading_order += 1

    log.info("Docling extracted %d blocks", len(blocks))
    return blocks


def _safe_text(item: Any) -> str:
    t = getattr(item, "text", None)
    if t is None:
        t = getattr(item, "orig", "")
    return str(t or "")


def _get_prov(item: Any) -> tuple[int, tuple[float, float, float, float]]:
    prov = getattr(item, "prov", None)
    if not prov:
        return 0, (0.0, 0.0, 0.0, 0.0)
    p = prov[0]
    page = int(getattr(p, "page_no", 0) or getattr(p, "page", 0) or 0)
    bbox_obj = getattr(p, "bbox", None)
    if bbox_obj is None:
        return page, (0.0, 0.0, 0.0, 0.0)
    bbox = (
        float(getattr(bbox_obj, "l", 0.0)),
        float(getattr(bbox_obj, "t", 0.0)),
        float(getattr(bbox_obj, "r", 0.0)),
        float(getattr(bbox_obj, "b", 0.0)),
    )
    return page, bbox


def _table_to_html(item: Any) -> str | None:
    for fn_name in ("export_to_html", "to_html"):
        fn = getattr(item, fn_name, None)
        if fn is None:
            continue
        try:
            return fn()
        except TypeError:
            try:
                return fn(doc=None)
            except Exception:
                continue
        except Exception:
            continue
    data = getattr(item, "data", None)
    if data is not None and hasattr(data, "to_html"):
        try:
            return data.to_html()
        except Exception:
            return None
    return None


def _table_to_text(item: Any) -> str:
    fn = getattr(item, "export_to_markdown", None)
    if fn is not None:
        try:
            return fn()
        except Exception:
            pass
    return _safe_text(item) or "[table]"


def _safe_caption(item: Any) -> str | None:
    captions = getattr(item, "captions", None)
    if not captions:
        return None
    parts: list[str] = []
    for c in captions:
        if hasattr(c, "text"):
            parts.append(c.text)
        elif hasattr(c, "resolve"):
            try:
                parts.append(c.resolve().text)  # CaptionRef
            except Exception:
                continue
        else:
            parts.append(str(c))
    out = " ".join(parts).strip()
    return out or None


def _anchor_for(section_stack: list[tuple[int, str]]) -> str | None:
    if not section_stack:
        return None
    label = section_stack[-1][1]
    slug = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
    return slug[:64] or None
