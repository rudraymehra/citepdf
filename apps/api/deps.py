"""Shared FastAPI dependencies — singletons for embedder, reranker, qdrant, anthropic."""

from __future__ import annotations

from functools import lru_cache

import anthropic

from packages.core.settings import Settings, get_settings
from packages.ingest.embedder import get_embedder
from packages.ingest.index import get_qdrant
from packages.retrieve.rerank import get_reranker


@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    s = get_settings()
    return anthropic.Anthropic(api_key=s.anthropic_api_key)


def warmup() -> None:
    """Eagerly load heavy ML models at API boot so first request isn't slow."""
    get_embedder()
    get_reranker()
    get_qdrant()
    get_anthropic_client()


__all__ = [
    "Settings",
    "get_settings",
    "get_embedder",
    "get_qdrant",
    "get_reranker",
    "get_anthropic_client",
    "warmup",
]
