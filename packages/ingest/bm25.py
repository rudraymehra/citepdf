"""Classical BM25 sparse vectors via FastEmbed's Qdrant/bm25 model.

This is the *third* retrieval signal alongside:
  - BGE-M3 dense (semantic)
  - BGE-M3 lexical (learned sparse, SPLADE-like)
  - **BM25 (this module)** — classical TF-IDF, what Anthropic's
    Contextual Retrieval cookbook actually uses for the "BM25" leg.

We embed (contextual_header + chunk_text) so the BM25 channel benefits from
the same context augmentation as the dense channel. At query time, the user
query is tokenized into the same space.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import TypedDict

log = logging.getLogger(__name__)


class SparseEntry(TypedDict):
    indices: list[int]
    values: list[float]


class BM25Embedder:
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        log.info("Loading FastEmbed BM25 (%s)…", model_name)
        # Embed (document) and query_embed (query) use the same tokenizer here.
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> list[SparseEntry]:
        if not texts:
            return []
        out: list[SparseEntry] = []
        # FastEmbed returns generator of SparseEmbedding objects with .indices / .values
        for emb in self.model.embed(texts, batch_size=batch_size):
            out.append(_to_entry(emb))
        return out

    def embed_query(self, query: str) -> SparseEntry:
        for emb in self.model.query_embed([query]):
            return _to_entry(emb)
        return {"indices": [], "values": []}


def _to_entry(emb) -> SparseEntry:
    indices = [int(i) for i in getattr(emb, "indices", [])]
    values: list[float] = []
    for v in getattr(emb, "values", []):
        f = float(v)
        if not math.isfinite(f) or f == 0.0:
            continue
        values.append(f)
    # Re-align indices/values length if filtering happened. FastEmbed values
    # are already finite in practice; this is defense in depth.
    indices = indices[: len(values)]
    return {"indices": indices, "values": values}


@lru_cache(maxsize=1)
def get_bm25_embedder() -> BM25Embedder:
    return BM25Embedder()
