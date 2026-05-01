"""BGE-M3 embedder — emits dense (1024-dim) + sparse (lexical-weights) per text.

BGE-M3 is uniquely useful here: it produces a dense vector AND a learned
sparse representation in a single forward pass. The sparse output is
SPLADE-like (learned per-token weights), so we can drive Qdrant hybrid
retrieval (dense + sparse) with one model and zero cost beyond a single
~1.2 GB model download.

Numerical-stability notes:
- We default ``use_fp16`` to True only when a real CUDA GPU is present.
  On Apple Silicon and CPU, PyTorch's fp16 emulation can produce NaN on
  short or boilerplate inputs (e.g. a stub figure caption), and that
  single bad row poisons UMAP/GMM downstream. Use FORCE_FP16=1 or
  FORCE_FP32=1 env vars to override the auto-detect.
- After encoding, we replace any non-finite (NaN/inf) row with a
  zero vector and log a warning. Defense in depth — RAPTOR and
  Qdrant should never see a non-finite value.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TypedDict

import numpy as np

log = logging.getLogger(__name__)


class SparseEntry(TypedDict):
    indices: list[int]
    values: list[float]


def _should_use_fp16() -> bool:
    """fp16 is safe on real CUDA GPUs and unsafe on CPU / Apple Silicon."""
    force = os.getenv("FORCE_FP16", "").strip()
    if force == "1":
        return True
    if os.getenv("FORCE_FP32", "").strip() == "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class BGEM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool | None = None):
        from FlagEmbedding import BGEM3FlagModel

        if use_fp16 is None:
            use_fp16 = _should_use_fp16()
        log.info(
            "Loading BGE-M3 (%s, fp16=%s) — first run downloads ~1.2GB",
            model_name,
            use_fp16,
        )
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.use_fp16 = use_fp16
        self.dim = 1024

    def embed_texts(
        self, texts: list[str], batch_size: int = 8, max_length: int = 8192
    ) -> tuple[np.ndarray, list[SparseEntry]]:
        """Embed a list of texts. Returns (dense, sparse_entries).

        Guarantees: every row of `dense` is finite; sparse entries omit any
        non-finite weights. Empty/whitespace-only inputs are replaced with a
        placeholder so the model never sees an empty token sequence."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32), []

        safe_texts = [t if (t and t.strip()) else "[empty]" for t in texts]

        out = self.model.encode(
            safe_texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)

        bad_mask = ~np.isfinite(dense).all(axis=1)
        n_bad = int(bad_mask.sum())
        if n_bad:
            log.warning(
                "BGE-M3 produced %d/%d rows with NaN/inf — replacing with zeros "
                "(consider FORCE_FP32=1 if this happens often)",
                n_bad,
                len(dense),
            )
            dense[bad_mask] = 0.0

        sparse_dicts = out["lexical_weights"]
        sparse = [_sparse_dict_to_entry(d) for d in sparse_dicts]
        return dense, sparse

    def embed_query(self, query: str) -> tuple[np.ndarray, SparseEntry]:
        dense, sparse = self.embed_texts([query], batch_size=1, max_length=512)
        return dense[0], sparse[0]


def _sparse_dict_to_entry(d: dict) -> SparseEntry:
    """BGE-M3 returns a dict {token_id: weight}. Convert to (indices, values).

    Drops zero-weight entries (Qdrant doesn't need them) and silently
    skips NaN/inf weights (defense in depth)."""
    import math

    indices: list[int] = []
    values: list[float] = []
    for k, v in d.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == 0.0 or not math.isfinite(f):
            continue
        try:
            indices.append(int(k))
        except (ValueError, TypeError):
            continue
        values.append(f)
    return {"indices": indices, "values": values}


@lru_cache(maxsize=1)
def get_embedder() -> BGEM3Embedder:
    return BGEM3Embedder()
