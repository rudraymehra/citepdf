"""Hybrid retrieval (dense + sparse) with RRF fusion + bge-reranker-v2-m3,
plus RAPTOR collapsed-tree traversal and expand-to-L0 for citation fidelity.

Phase 2 flow:
  embed(query) -> Qdrant Query API with two prefetches (dense, sparse)
  -> RRF fusion (k=60) -> top TOPK_PREFETCH (50)
  -> bge-reranker-v2-m3 cross-encoder -> top TOPK_FINAL (8)
  -> expand any L>0 hit into its underlying L0 leaves (so citations always
     point to citable leaf pages, never summary nodes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from packages.core.settings import get_settings
from packages.ingest.bm25 import get_bm25_embedder
from packages.ingest.embedder import get_embedder
from packages.ingest.index import (
    BM25_VEC,
    DENSE_VEC,
    SPARSE_VEC,
    collection_name,
    fetch_chunks_by_id,
    get_qdrant,
)
from packages.retrieve.rerank import get_reranker

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    rerank_score: float
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def page_start(self) -> int:
        return int(self.payload.get("page_start", 0))

    @property
    def page_end(self) -> int:
        return int(self.payload.get("page_end", 0))

    @property
    def section_label(self) -> str:
        return self.payload.get("section_label", "") or ""

    @property
    def section_anchor(self) -> str | None:
        return self.payload.get("section_anchor") or None

    @property
    def level(self) -> int:
        return int(self.payload.get("level", 0))


def retrieve(
    doc_id: str,
    query: str,
    top_k: int | None = None,
    prefetch_k: int | None = None,
    filters: dict[str, Any] | None = None,
    expand_to_leaves: bool = True,
) -> list[RetrievedChunk]:
    """Hybrid retrieval. Returns top_k L0-leaf chunks (after expand) by default."""
    import time as _time
    t = _time.perf_counter()

    s = get_settings()
    top_k = top_k or s.topk_final
    prefetch_k = prefetch_k or s.topk_prefetch

    embedder = get_embedder()
    dense_vec, sparse_entry = embedder.embed_query(query)
    t_embed = _time.perf_counter() - t
    t = _time.perf_counter()
    bm25 = get_bm25_embedder().embed_query(query)
    t_bm25 = _time.perf_counter() - t
    qdrant_filter = _build_filter(doc_id, filters)

    client = get_qdrant()
    coll = collection_name(doc_id)

    prefetches = [
        Prefetch(
            query=dense_vec.tolist(),
            using=DENSE_VEC,
            limit=prefetch_k,
            filter=qdrant_filter,
        ),
        Prefetch(
            query=SparseVector(
                indices=sparse_entry["indices"],
                values=sparse_entry["values"],
            ),
            using=SPARSE_VEC,
            limit=prefetch_k,
            filter=qdrant_filter,
        ),
    ]
    # Add BM25 prefetch only if the query produced any tokens (very short
    # queries can produce empty BM25 vectors).
    if bm25.get("indices"):
        prefetches.append(
            Prefetch(
                query=SparseVector(
                    indices=bm25["indices"],
                    values=bm25["values"],
                ),
                using=BM25_VEC,
                limit=prefetch_k,
                filter=qdrant_filter,
            )
        )

    t = _time.perf_counter()
    response = client.query_points(
        collection_name=coll,
        prefetch=prefetches,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=prefetch_k,
        with_payload=True,
    )
    candidates = list(response.points)
    t_qdrant = _time.perf_counter() - t
    if not candidates:
        log.info("retrieve(%s): empty candidate pool", query[:60])
        return []

    # Expand any L>0 (summary) to its L0 children so the reranker scores actual
    # citable text. Keep the summary-derived candidates de-duplicated by chunk_id.
    t = _time.perf_counter()
    if expand_to_leaves:
        candidates = _expand_to_leaves(client, coll, candidates)
    t_expand = _time.perf_counter() - t

    # Cross-encoder rerank
    t = _time.perf_counter()
    reranker = get_reranker()
    docs = [p.payload.get("text", "") if p.payload else "" for p in candidates]
    rerank_scores = reranker.score(query, docs)
    t_rerank = _time.perf_counter() - t

    log.info(
        "retrieve.timings: embed=%.2fs bm25=%.2fs qdrant=%.2fs expand=%.2fs rerank=%.2fs (n=%d)",
        t_embed, t_bm25, t_qdrant, t_expand, t_rerank, len(candidates),
    )

    enriched: list[RetrievedChunk] = []
    for hit, rscore in zip(candidates, rerank_scores):
        enriched.append(
            RetrievedChunk(
                chunk_id=str(hit.id),
                score=float(hit.score or 0.0),
                rerank_score=float(rscore),
                payload=dict(hit.payload or {}),
            )
        )
    enriched.sort(key=lambda c: c.rerank_score, reverse=True)
    return enriched[:top_k]


def _expand_to_leaves(
    client: Any,
    coll: str,
    candidates: list[Any],
) -> list[Any]:
    """For every non-leaf candidate, replace it with its L0 children. Dedupe."""
    # Build a unique set of chunk ids: leaves stay; summaries contribute their child_ids.
    leaf_ids: list[str] = []
    leaf_id_set: set[str] = set()
    summary_child_ids_to_fetch: list[str] = []

    for hit in candidates:
        payload = hit.payload or {}
        level = int(payload.get("level", 0))
        chid = str(hit.id)
        if level == 0:
            if chid not in leaf_id_set:
                leaf_ids.append(chid)
                leaf_id_set.add(chid)
        else:
            for c in payload.get("child_ids", []) or []:
                if c and c not in leaf_id_set:
                    summary_child_ids_to_fetch.append(c)
                    leaf_id_set.add(c)

    # Fetch summary children
    fetched = []
    if summary_child_ids_to_fetch:
        # cap to a reasonable batch to avoid huge fetches
        fetched = fetch_chunks_by_id(client, coll, summary_child_ids_to_fetch[:200])

    # Build a fake "hit-like" object list: keep original L0 hits; convert fetched leaves
    # into a payload-only structure. The reranker only needs payload.text + id.
    out: list[Any] = []
    seen: set[str] = set()
    for hit in candidates:
        if int((hit.payload or {}).get("level", 0)) == 0 and str(hit.id) not in seen:
            out.append(hit)
            seen.add(str(hit.id))
    for f in fetched:
        chid = f["chunk_id"]
        if chid in seen:
            continue
        out.append(_FakeHit(id=chid, score=0.0, payload=f))
        seen.add(chid)
    return out


@dataclass
class _FakeHit:
    """Minimal duck-typed hit object for fetched leaves (matches Qdrant's ScoredPoint shape)."""

    id: str
    score: float
    payload: dict


def _build_filter(doc_id: str, filters: dict[str, Any] | None) -> Filter:
    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if filters:
        for k, v in filters.items():
            must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=must)
