"""Out-of-scope detection.

Phase 1 strategy (per architecture doc §6.5, cheap stage only):
  cosine(embed(query), doc_centroid) < OOS_CENTROID_THRESHOLD  → refuse.

Plus a retrieval-side guard: if top-1 rerank score < OOS_TOP1_THRESHOLD,
also refuse.

Phase 2 adds the LLM-judge expensive stage for borderline (0.30–0.45) cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from packages.core.settings import get_settings
from packages.ingest.embedder import get_embedder
from packages.ingest.index import read_doc_meta
from packages.retrieve.retriever import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class OOSResult:
    in_scope: bool
    centroid_sim: float
    reason: str = ""


def in_scope_query(doc_id: str, query: str) -> OOSResult:
    s = get_settings()
    meta = read_doc_meta(doc_id)
    if meta is None or "centroid" not in meta:
        # No centroid persisted — fall through and rely on retrieval-side guard.
        return OOSResult(in_scope=True, centroid_sim=1.0, reason="no_centroid")

    centroid = np.array(meta["centroid"], dtype=np.float32)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

    embedder = get_embedder()
    q_dense, _ = embedder.embed_query(query)
    q_norm = q_dense / (np.linalg.norm(q_dense) + 1e-12)
    sim = float(np.dot(centroid_norm, q_norm))

    if sim < s.oos_centroid_threshold:
        return OOSResult(in_scope=False, centroid_sim=sim, reason="centroid_low")
    return OOSResult(in_scope=True, centroid_sim=sim, reason="")


def retrieval_passes_threshold(chunks: list[RetrievedChunk]) -> bool:
    """Second-stage guard: refuse if no retrieved chunk crosses the rerank threshold."""
    s = get_settings()
    if not chunks:
        return False
    return chunks[0].rerank_score >= s.oos_top1_threshold
