"""FastAPI app entry — wires routes_upload + routes_chat + health."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rich.logging import RichHandler

from apps.api import routes_chat, routes_upload
from apps.api.deps import warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up models (BGE-M3 + reranker) — first run downloads ~1.8GB")
    try:
        warmup()
        log.info("Warmup complete")
    except Exception:
        log.exception("Warmup failed; first request will pay the model load cost")
    yield


app = FastAPI(
    title="PDF-Constrained Conversational RAG Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(routes_upload.router)
app.include_router(routes_chat.router)
