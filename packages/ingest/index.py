"""Qdrant collection setup + chunk upsert + per-doc metadata sidecar.

Per-doc collection name: ``doc_{doc_id}_chunks``. One named dense vector
("dense", 1024-dim cosine) and one sparse vector ("sparse", BGE-M3 lexical
weights, no IDF modifier — they are already learned weights).

Per-doc sidecar at ``{data_dir}/meta_{doc_id}.json`` stores: centroid vector
(mean of all dense vectors, used for OOS detection), language list, page
range, summary fields populated in Phase 2.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from packages.core.schema import Chunk
from packages.core.settings import get_settings

log = logging.getLogger(__name__)

DENSE_VEC = "dense"
SPARSE_VEC = "sparse"  # BGE-M3 lexical weights (learned, SPLADE-like)
BM25_VEC = "bm25"  # Classical BM25 from FastEmbed (Qdrant/bm25)
DENSE_DIM = 1024


class DocMeta(TypedDict, total=False):
    doc_id: str
    n_chunks: int
    pages: list[int]
    languages: list[str]
    centroid: list[float]
    pdf_path: str  # absolute filesystem path to the original PDF


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    s = get_settings()
    # check_compatibility=False suppresses a noisy warning when the pinned
    # qdrant-client (e.g. 1.17.x) is more than one minor ahead of the local
    # Docker image (e.g. 1.12.4). API surface we use (collections + points
    # + Query API + sparse vectors) is stable across these versions.
    return QdrantClient(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key,
        prefer_grpc=False,
        check_compatibility=False,
    )


def collection_name(doc_id: str) -> str:
    safe = doc_id.replace("-", "_")
    return f"doc_{safe}_chunks"


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(collection_name=name):
        log.info("Collection %s already exists; reusing", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VEC: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            # BGE-M3 learned sparse — used as-is, no IDF (already weighted).
            SPARSE_VEC: SparseVectorParams(),
            # Classical BM25 with server-side IDF modifier (FastEmbed Qdrant/bm25
            # output is raw TF; Qdrant computes IDF from the collection).
            BM25_VEC: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            ),
        },
    )
    # Index the most-filtered payload fields
    for field, schema in [
        ("doc_id", "keyword"),
        ("level", "integer"),
        ("page_start", "integer"),
        ("page_end", "integer"),
        ("section_anchor", "keyword"),
        ("language", "keyword"),
    ]:
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
        except Exception as e:
            log.debug("payload index %s already exists or failed: %s", field, e)
    log.info("Created collection %s", name)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense: np.ndarray,
    sparse: list[dict],
    bm25: list[dict] | None = None,
    batch_size: int = 64,
) -> None:
    points: list[PointStruct] = []
    for i, chunk in enumerate(chunks):
        sp = sparse[i]
        bm = (bm25 or [])[i] if bm25 and i < len(bm25) else None
        payload = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "level": chunk.level,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "pages": chunk.pages,
            "section_anchor": chunk.section_anchor or "",
            "section_label": chunk.display_section,
            "section_paths": [list(p) for p in chunk.section_paths],
            "language": chunk.language,
            "block_types": chunk.block_types,
            "source_block_ids": chunk.source_block_ids,
            "child_ids": chunk.child_ids,
            "text": chunk.text,
            "html_table": chunk.html_table,
            "image_ref": chunk.image_ref,
            "image_caption": chunk.image_caption,
            "contextual_header": chunk.contextual_header,
            "block_bboxes": [list(b) for b in chunk.block_bboxes],
        }
        vec: dict = {
            DENSE_VEC: dense[i].tolist(),
            SPARSE_VEC: SparseVector(
                indices=sp["indices"],
                values=sp["values"],
            ),
        }
        if bm and bm.get("indices"):
            vec[BM25_VEC] = SparseVector(
                indices=bm["indices"],
                values=bm["values"],
            )
        points.append(PointStruct(id=chunk.chunk_id, vector=vec, payload=payload))
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i : i + batch_size])
    log.info("Upserted %d points to %s", len(points), collection)


def write_doc_meta(
    doc_id: str,
    chunks: list[Chunk],
    dense: np.ndarray,
    pdf_path: str | None = None,
) -> Path:
    s = get_settings()
    centroid = dense.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    meta: DocMeta = {
        "doc_id": doc_id,
        "n_chunks": len(chunks),
        "pages": sorted({p for c in chunks for p in c.pages}),
        "languages": sorted({c.language for c in chunks}),
        "centroid": centroid.astype(float).tolist(),
    }
    if pdf_path:
        meta["pdf_path"] = pdf_path
    path = Path(s.data_dir) / f"meta_{doc_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(meta, f)
    return path


def read_doc_meta(doc_id: str) -> DocMeta | None:
    s = get_settings()
    path = Path(s.data_dir) / f"meta_{doc_id}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def fetch_chunks_by_id(
    client: QdrantClient,
    collection: str,
    chunk_ids: list[str],
) -> list[dict]:
    """Fetch a set of points by id and return their payloads."""
    if not chunk_ids:
        return []
    points = client.retrieve(
        collection_name=collection,
        ids=chunk_ids,
        with_payload=True,
        with_vectors=False,
    )
    return [dict(p.payload or {}) | {"chunk_id": str(p.id)} for p in points]
