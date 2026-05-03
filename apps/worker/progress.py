"""Redis-backed ingest-progress reporting.

Each ingest run writes its current state to a Redis hash keyed by doc_id:
    HSET ingest:{doc_id} status <pending|running|done|error>
                          phase <parsing|chunking|...|done>
                          fraction <0.0..1.0>
                          message <human label>
                          updated_at <iso8601>
                          ...optional metrics

Both API and frontend can read the same hash to render a progress bar.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import redis.asyncio as aioredis
import redis as syncredis

from packages.core.settings import get_settings

log = logging.getLogger(__name__)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def progress_key(doc_id: str) -> str:
    return f"ingest:{doc_id}"


def make_sync_progress_writer(doc_id: str):
    """Return a callable progress(phase, fraction, **extras) backed by a sync Redis client."""
    s = get_settings()
    client = syncredis.from_url(s.redis_url, decode_responses=True)
    key = progress_key(doc_id)

    def write(phase: str, fraction: float, **extras: Any) -> None:
        mapping = {
            "status": extras.pop("status", "running"),
            "phase": phase,
            "fraction": f"{fraction:.4f}",
            "updated_at": _now(),
            "doc_id": doc_id,
        }
        for k, v in extras.items():
            mapping[k] = v if isinstance(v, str) else json.dumps(v)
        try:
            client.hset(key, mapping=mapping)
            client.expire(key, 60 * 60 * 6)  # 6 hours
        except Exception as e:
            log.warning("progress write failed: %s", e)

    return write


async def read_progress_async(doc_id: str) -> dict[str, Any]:
    s = get_settings()
    client = aioredis.from_url(s.redis_url, decode_responses=True)
    try:
        raw = await client.hgetall(progress_key(doc_id))
    finally:
        await client.aclose()
    if not raw:
        return {"status": "unknown", "phase": "", "fraction": 0.0, "doc_id": doc_id}
    return _normalize(raw)


def read_progress_sync(doc_id: str) -> dict[str, Any]:
    s = get_settings()
    client = syncredis.from_url(s.redis_url, decode_responses=True)
    raw = client.hgetall(progress_key(doc_id))
    if not raw:
        return {"status": "unknown", "phase": "", "fraction": 0.0, "doc_id": doc_id}
    return _normalize(raw)


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(raw)
    if "fraction" in out:
        try:
            out["fraction"] = float(out["fraction"])
        except (TypeError, ValueError):
            out["fraction"] = 0.0
    for k in ("summary", "error_detail"):
        if k in out and isinstance(out[k], str):
            try:
                out[k] = json.loads(out[k])
            except (json.JSONDecodeError, TypeError):
                pass
    return out
