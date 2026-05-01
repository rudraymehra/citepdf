"""POST /chat — SSE streaming endpoint.

Wire format (each line is one SSE event):

    event: text
    data: {"text": "..."}

    event: citation
    data: {"chunk_id": "...", "page_start": 3, "page_end": 3,
           "section_label": "§3.2", "cited_text": "..."}

    event: refusal
    data: {"text": "I cannot answer this from the provided document."}

    event: done
    data: {}

Phase 1 emits one `text` event with the full answer (Anthropic non-streaming).
Phase 2 swaps to true token streaming via Anthropic's stream API.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from packages.agent.agent import chat
from packages.core.schema import ChatRequest

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def event_stream() -> AsyncIterator[dict]:
        # agent.chat is a sync generator — run it in a threadpool so we don't
        # block the event loop. iterate_in_threadpool handles step-by-step.
        sync_iter = chat(
            doc_id=req.doc_id,
            user_message=req.query,
            history=req.history,
            mode=req.mode,
        )
        async for event in iterate_in_threadpool(sync_iter):
            payload = _serialize(event)
            yield {"event": event.kind, "data": json.dumps(payload)}

    return EventSourceResponse(event_stream())


def _serialize(event) -> dict:
    if event.kind == "text":
        return {"text": event.text or ""}
    if event.kind == "refusal":
        return {"text": event.text or ""}
    if event.kind == "warning":
        return {"text": event.text or ""}
    if event.kind == "error":
        return {"error": event.error or "unknown error"}
    if event.kind == "citation" and event.citation is not None:
        c = event.citation
        return {
            "chunk_id": c.chunk_id,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_anchor": c.section_anchor,
            "section_label": c.section_label,
            "cited_text": c.cited_text,
            "block_types": c.block_types,
            "image_ref": c.image_ref,
            "render": c.render(),
            "bboxes": [
                {"page": p, "bbox": list(b)} for (p, b) in (c.bboxes or [])
            ],
        }
    return {}
