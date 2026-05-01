"""bge-reranker-v2-m3 cross-encoder for top-N reranking.

Multilingual (100+ languages incl. Hindi/Tamil/Telugu/Bengali/etc.). ~560 MB
model. Runs on CPU acceptably for top-50 candidates per query.

fp16 auto-detect: same logic as BGE-M3 — only enable on real CUDA GPUs to
avoid NaN scores on Apple Silicon CPU. Override with FORCE_FP16=1 / FORCE_FP32=1.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from packages.ingest.embedder import _should_use_fp16

log = logging.getLogger(__name__)


class BGEReranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool | None = None
    ):
        from FlagEmbedding import FlagReranker

        if use_fp16 is None:
            use_fp16 = _should_use_fp16()
        log.info(
            "Loading bge-reranker-v2-m3 (fp16=%s) — first run downloads ~560MB", use_fp16
        )
        self.model = FlagReranker(model_name, use_fp16=use_fp16)
        self.use_fp16 = use_fp16

    def score(self, query: str, docs: list[str], batch_size: int = 16) -> list[float]:
        if not docs:
            return []
        pairs = [[query, d if (d and d.strip()) else "[empty]"] for d in docs]
        scores = self.model.compute_score(pairs, batch_size=batch_size, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        cleaned: list[float] = []
        n_bad = 0
        for s in scores:
            f = float(s)
            if not math.isfinite(f):
                n_bad += 1
                f = 0.0
            cleaned.append(f)
        if n_bad:
            log.warning(
                "Reranker produced %d/%d non-finite scores — replacing with 0.0",
                n_bad,
                len(scores),
            )
        return cleaned


@lru_cache(maxsize=1)
def get_reranker() -> BGEReranker:
    return BGEReranker()
