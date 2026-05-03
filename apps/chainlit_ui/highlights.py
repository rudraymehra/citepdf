"""Render PDF pages with bbox-highlighted citation overlays.

Given a PDF path, page number, and a list of bboxes in PDF-point coordinates,
produce a PNG with semi-transparent yellow rectangles drawn over the cited
regions. Used by the Chainlit citation side-panel.

PDF coords (Docling/PyMuPDF): (l, t, r, b) in points where the y-axis
origin is at the top of the page (PyMuPDF convention). Scale by the
rendered DPI to map to image pixels.

Outputs are cached on disk by content hash to avoid re-rendering across
multiple citations of the same region.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

DPI = int(os.getenv("HIGHLIGHT_DPI", "150"))
CACHE_DIR = Path(os.getenv("HIGHLIGHT_CACHE_DIR", "./data/highlights"))
HIGHLIGHT_FILL = (255, 220, 0, 80)  # yellow @ 31% alpha
HIGHLIGHT_OUTLINE = (255, 165, 0, 255)  # orange outline
HIGHLIGHT_WIDTH = 3


def _meta_path(doc_id: str) -> Path:
    data_dir = Path(os.getenv("DATA_DIR", "./eval/data"))
    return data_dir / f"meta_{doc_id}.json"


def get_pdf_path(doc_id: str) -> str | None:
    """Read the PDF path stored in the doc meta sidecar (written at ingest)."""
    meta_path = _meta_path(doc_id)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    pdf_path = meta.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        return pdf_path
    return None


def render_highlight(
    pdf_path: str,
    page: int,
    bboxes: list[tuple[float, float, float, float]],
) -> Path | None:
    """Render `page` of `pdf_path` with `bboxes` highlighted. Returns PNG path.

    Cached by hash of (pdf_path, page, bboxes, DPI).
    """
    if not pdf_path or not bboxes or page <= 0:
        return None
    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageDraw
    except ImportError as e:
        log.warning("PyMuPDF/Pillow not available; skipping highlight render: %s", e)
        return None

    cache_key = hashlib.sha1(
        json.dumps([pdf_path, page, bboxes, DPI], sort_keys=True).encode()
    ).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{cache_key}.png"
    if out_path.exists():
        return out_path

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.warning("Could not open PDF %s: %s", pdf_path, e)
        return None

    if page < 1 or page > len(doc):
        doc.close()
        return None

    pdf_page = doc[page - 1]  # PyMuPDF is 0-indexed
    matrix = fitz.Matrix(DPI / 72.0, DPI / 72.0)
    pix = pdf_page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    page_w_pt = pdf_page.rect.width
    page_h_pt = pdf_page.rect.height
    scale_x = img.width / page_w_pt if page_w_pt > 0 else 1.0
    scale_y = img.height / page_h_pt if page_h_pt > 0 else 1.0

    for bbox in bboxes:
        l, t, r, b = bbox
        x0 = max(0, l * scale_x)
        y0 = max(0, t * scale_y)
        x1 = min(img.width, r * scale_x)
        y1 = min(img.height, b * scale_y)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=HIGHLIGHT_FILL,
            outline=HIGHLIGHT_OUTLINE,
            width=HIGHLIGHT_WIDTH,
        )

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    composited.save(out_path, format="PNG", optimize=True)
    doc.close()
    log.debug("Rendered highlight %s for page %d (%d bboxes)", out_path, page, len(bboxes))
    return out_path


def render_for_citation(
    doc_id: str,
    bboxes_with_pages: list[tuple[int, tuple[float, float, float, float]]],
) -> list[tuple[int, Path]]:
    """Group bboxes by page, render one image per cited page, return [(page, path)]."""
    pdf_path = get_pdf_path(doc_id)
    if not pdf_path:
        return []
    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for page, bbox in bboxes_with_pages:
        by_page.setdefault(int(page), []).append(tuple(bbox))
    out: list[tuple[int, Path]] = []
    for page, bbs in sorted(by_page.items()):
        path = render_highlight(pdf_path, page, bbs)
        if path is not None:
            out.append((page, path))
    return out
