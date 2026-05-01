"""POST /upload — enqueue an async ingest job and return immediately.
GET /upload/status/{doc_id} — poll Redis-backed progress.

The arq worker (apps.worker.worker.WorkerSettings) consumes the job and runs
the full ingest pipeline. Phase 1's sync /upload behavior is retained as
/upload/sync for the eval script and CLI parity."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from apps.worker.progress import make_sync_progress_writer, read_progress_async
from packages.core.schema import UploadResponse
from packages.core.settings import get_settings
from packages.ingest.ingest import ingest

router = APIRouter()
log = logging.getLogger(__name__)


class EnqueueResponse(BaseModel):
    doc_id: str
    job_id: str
    status_url: str


class StatusResponse(BaseModel):
    doc_id: str
    status: str
    phase: str = ""
    fraction: float = 0.0
    message: str = ""
    summary: dict[str, Any] | None = None
    error_detail: str | None = None


def _save_upload(file: UploadFile) -> tuple[str, Path]:
    s = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    doc_id = str(uuid.uuid4())
    dest_dir = Path(s.data_dir) / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    t0 = time.time()
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    log.info(
        "Saved %s (%.1f MB) in %.1fs",
        dest_path,
        dest_path.stat().st_size / 1e6,
        time.time() - t0,
    )
    return doc_id, dest_path


@router.post("/upload", response_model=EnqueueResponse)
async def upload(file: UploadFile = File(...)) -> EnqueueResponse:
    """Enqueue an async ingest job; return job_id + status URL immediately."""
    s = get_settings()
    doc_id, dest_path = _save_upload(file)
    # Mark queued in Redis even before the worker picks up the job
    make_sync_progress_writer(doc_id)("queued", 0.0, status="queued", message="Job enqueued")

    redis_settings = RedisSettings.from_dsn(s.redis_url)
    pool = await create_pool(redis_settings)
    try:
        job = await pool.enqueue_job("ingest_job", doc_id, str(dest_path))
    finally:
        await pool.aclose()
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to enqueue ingest job (already running?)")

    return EnqueueResponse(
        doc_id=doc_id,
        job_id=job.job_id,
        status_url=f"/upload/status/{doc_id}",
    )


@router.get("/upload/status/{doc_id}", response_model=StatusResponse)
async def upload_status(doc_id: str) -> StatusResponse:
    progress = await read_progress_async(doc_id)
    summary = progress.get("summary")
    return StatusResponse(
        doc_id=doc_id,
        status=progress.get("status", "unknown"),
        phase=progress.get("phase", ""),
        fraction=float(progress.get("fraction", 0.0)),
        message=progress.get("message", ""),
        summary=summary if isinstance(summary, dict) else None,
        error_detail=progress.get("error_detail") if isinstance(progress.get("error_detail"), str) else None,
    )


@router.post("/upload/sync", response_model=UploadResponse)
async def upload_sync(file: UploadFile = File(...)) -> UploadResponse:
    """Phase-1 fallback: sync ingest, useful for testing without a running worker."""
    doc_id, dest_path = _save_upload(file)
    try:
        summary = ingest(str(dest_path), doc_id=doc_id)
    except Exception as e:
        log.exception("Sync ingest failed for %s", dest_path)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
    return UploadResponse(
        doc_id=summary["doc_id"],
        n_blocks=summary["n_blocks"],
        n_chunks=summary["n_chunks"],
        pages=summary["pages"],
        languages=summary["languages"],
    )
