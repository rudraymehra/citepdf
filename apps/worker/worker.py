"""arq async ingestion worker.

Run with:

    uv run arq apps.worker.worker.WorkerSettings

The worker listens to Redis for `ingest_job` tasks. Each job runs the full
ingest pipeline (Docling parse → Contextual headers → BGE-M3 embed → RAPTOR
→ Qdrant upsert) and writes phase-by-phase progress to a Redis hash that
the API and frontends poll.
"""

from __future__ import annotations

import asyncio
import logging
import traceback

from arq.connections import RedisSettings
from rich.logging import RichHandler

from apps.worker.progress import make_sync_progress_writer
from packages.core.settings import get_settings
from packages.ingest.ingest import ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger(__name__)


async def ingest_job(ctx: dict, doc_id: str, pdf_path: str) -> dict:
    """arq task. Runs the sync ingest in a thread so we don't block the loop."""
    log.info("ingest_job start: doc_id=%s pdf=%s", doc_id, pdf_path)
    progress = make_sync_progress_writer(doc_id)
    progress("queued", 0.0, status="running", message="Worker received job")

    def run() -> dict:
        try:
            return ingest(pdf_path=pdf_path, doc_id=doc_id, progress=progress)
        except Exception as e:
            log.exception("ingest failed")
            progress(
                "error",
                1.0,
                status="error",
                message=f"{type(e).__name__}: {e}",
                error_detail=traceback.format_exc(limit=8),
            )
            raise

    return await asyncio.to_thread(run)


async def startup(ctx: dict) -> None:
    log.info("worker booting; warming heavy ML models…")
    # Eager-load heavy models so first job is fast.
    from apps.api.deps import warmup
    await asyncio.to_thread(warmup)
    log.info("worker ready")


async def shutdown(ctx: dict) -> None:
    log.info("worker shutting down")


class WorkerSettings:
    functions = [ingest_job]
    on_startup = startup
    on_shutdown = shutdown
    job_timeout = 60 * 30  # 30 minutes per ingest

    @classmethod
    def get_redis_settings(cls) -> RedisSettings:
        return RedisSettings.from_dsn(get_settings().redis_url)

    redis_settings = property(lambda self: WorkerSettings.get_redis_settings())  # type: ignore[assignment]


# arq looks for `redis_settings` on the class. We provide it as a class attr below
# (RedisSettings.from_dsn evaluated at import time).
WorkerSettings.redis_settings = RedisSettings.from_dsn(get_settings().redis_url)  # type: ignore[assignment]
