"""End-to-end ingest orchestrator + CLI.

    pdf-ingest /path/to/file.pdf [--doc-id DOC_ID]
    # or:
    python -m packages.ingest.ingest /path/to/file.pdf

Pipeline: parse (Docling) → language tag → chunk → embed (BGE-M3 dense+sparse)
→ Qdrant upsert → meta sidecar (centroid for OOS).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from packages.core.schema import Chunk
from packages.core.settings import get_settings
from packages.ingest.bm25 import get_bm25_embedder
from packages.ingest.chunker import chunk_blocks
from packages.ingest.contextual import add_contextual_headers
from packages.ingest.embedder import get_embedder
from packages.ingest.index import (
    collection_name,
    ensure_collection,
    get_qdrant,
    upsert_chunks,
    write_doc_meta,
)
from packages.ingest.lang import detect_lang, dominant_language
from packages.ingest.parsers import parse_pdf
from packages.ingest.raptor import build_raptor_tree

console = Console()


def ingest(
    pdf_path: str,
    doc_id: str | None = None,
    progress: Callable[..., None] | None = None,
) -> dict:
    """Run the full ingest pipeline. Optional `progress(phase, fraction, **extras)`
    callback (Redis-backed when called from the worker; None for CLI)."""
    s = get_settings()
    doc_id = doc_id or str(uuid.uuid4())
    pdf_path = str(Path(pdf_path).expanduser().resolve())
    p = progress or (lambda *a, **kw: None)

    console.rule(f"[bold cyan]Ingest[/bold cyan] {pdf_path}")
    console.print(f"  doc_id = [bold]{doc_id}[/bold]")
    console.print(f"  qdrant = {s.qdrant_url}")

    p("parsing", 0.05, message="Parsing PDF with Docling")
    t0 = time.time()
    blocks = parse_pdf(pdf_path, doc_id)
    console.print(f"  parsed: {len(blocks)} blocks in {time.time() - t0:.1f}s")

    p("language", 0.20, message="Detecting language per block", n_blocks=len(blocks))
    t0 = time.time()
    for b in blocks:
        b.language = detect_lang(b.text)
    console.print(f"  language tagged in {time.time() - t0:.1f}s")

    p("chunking", 0.25, message="Chunking layout-aware")
    t0 = time.time()
    chunks: list[Chunk] = chunk_blocks(blocks, doc_id)
    for c in chunks:
        if not c.language or c.language == "":
            c.language = dominant_language(
                [b.language for b in blocks if b.block_id in c.source_block_ids]
            )
    console.print(f"  chunked: {len(chunks)} chunks in {time.time() - t0:.1f}s")

    if not chunks:
        console.print("[red]No chunks produced — aborting[/red]")
        p("done", 1.0, status="error", message="No chunks produced")
        return {"doc_id": doc_id, "n_blocks": len(blocks), "n_chunks": 0, "pages": 0, "languages": []}

    if s.enable_contextual_headers:
        p("contextual_headers", 0.30, message="Generating contextual headers (cached prefix)")
        t0 = time.time()
        add_contextual_headers(chunks, blocks)
        console.print(f"  contextual headers in {time.time() - t0:.1f}s")

    p("embedding_leaves", 0.55, message="Embedding leaves with BGE-M3")
    t0 = time.time()
    embedder = get_embedder()
    texts_to_embed = [
        (c.contextual_header + "\n" + c.text).strip() if c.contextual_header else c.text
        for c in chunks
    ]
    leaf_dense, leaf_sparse = embedder.embed_texts(texts_to_embed, batch_size=8)
    console.print(f"  embedded leaves ({leaf_dense.shape[0]} × {leaf_dense.shape[1]}) in {time.time() - t0:.1f}s")

    if s.enable_raptor:
        p("raptor", 0.70, message="Building RAPTOR tree (UMAP+GMM, summaries)")
        t0 = time.time()
        bundle = build_raptor_tree(chunks, leaf_dense, leaf_sparse, embedder)
        console.print(
            f"  RAPTOR built ({len(bundle.chunks)} chunks total, "
            f"{len(bundle.chunks) - len(chunks)} tree nodes) in {time.time() - t0:.1f}s"
        )
        all_chunks = bundle.chunks
        all_dense = bundle.dense
        all_sparse = bundle.sparse
    else:
        all_chunks = chunks
        all_dense = leaf_dense
        all_sparse = leaf_sparse

    # Phase 2: classical BM25 sparse vectors over (header + chunk) text.
    # Anthropic's "Contextual Retrieval" includes contextual BM25 as the
    # 3rd channel alongside contextual embeddings + dense.
    p("bm25", 0.85, message="Computing classical BM25 sparse vectors")
    t0 = time.time()
    bm25_embedder = get_bm25_embedder()
    bm25_texts = [
        (c.contextual_header + "\n" + c.text).strip() if c.contextual_header else c.text
        for c in all_chunks
    ]
    bm25_vectors = bm25_embedder.embed_texts(bm25_texts, batch_size=16)
    console.print(f"  BM25 vectors ({len(bm25_vectors)}) in {time.time() - t0:.1f}s")

    p("indexing", 0.92, message="Upserting to Qdrant", n_chunks=len(all_chunks))
    t0 = time.time()
    client = get_qdrant()
    coll = collection_name(doc_id)
    ensure_collection(client, coll)
    upsert_chunks(client, coll, all_chunks, all_dense, all_sparse, bm25=bm25_vectors)
    meta_path = write_doc_meta(doc_id, chunks, leaf_dense, pdf_path=pdf_path)
    console.print(f"  upserted to {coll} in {time.time() - t0:.1f}s")
    console.print(f"  meta sidecar: {meta_path}")

    pages = sorted({p_ for c in chunks for p_ in c.pages})
    languages = sorted({c.language for c in chunks})
    summary = {
        "doc_id": doc_id,
        "n_blocks": len(blocks),
        "n_chunks": len(chunks),
        "n_chunks_total_with_tree": len(all_chunks),
        "pages": len(pages),
        "languages": languages,
        "collection": coll,
    }
    console.print("[green]Done[/green]", summary)
    p("done", 1.0, status="done", summary=summary)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG index")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--doc-id", default=None, help="Override doc_id (UUID by default)")
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        console.print(f"[red]File not found:[/red] {args.pdf_path}")
        sys.exit(1)

    ingest(args.pdf_path, args.doc_id)


if __name__ == "__main__":
    main()
