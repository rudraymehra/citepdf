"""Chainlit UI — chats with the FastAPI backend.

Run:
    uv run chainlit run apps/chainlit_ui/app.py --port 8502

Features:
    - ChatGPT-style left sidebar with all past threads (SQLAlchemy + SQLite)
    - File upload via cl.AskFileMessage on chat start; polls /upload/status until done
    - Chat profiles for `instant` (~3s p50) and `deep` (HyDE + multi-query + verifier)
    - SSE streaming from /chat with citations rendered as side-panel Elements
    - Token-by-token streaming (first word in <1s)
    - Per-session doc_id + chat history; threads + messages persisted to SQLite

Persistence: enabled when CHAINLIT_DB_URL is set in env (default sqlite path
under ./data/chainlit.db). Auth is single-user "reviewer" identity for the
demo — every visitor sees the same threads, no login. For multi-tenant
deploy, replace the header_auth_callback with real auth.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import chainlit as cl
import httpx

# Use a path-aware import that works whether Chainlit launches from the
# project root (cwd=/Users/rudraym/pdf) or from this file's directory.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from apps.chainlit_ui.highlights import render_for_citation  # noqa: E402

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REFUSAL_STRING = os.getenv("REFUSAL_STRING", "I cannot answer this from the provided document.")
CHAINLIT_DB_URL = os.getenv("CHAINLIT_DB_URL", "")


# ---------------------------------------------------------------------------
# Persistence + auth — gives the ChatGPT-style sidebar
# ---------------------------------------------------------------------------

if CHAINLIT_DB_URL:
    # Ensure the SQLite parent dir exists for the file-based default.
    if CHAINLIT_DB_URL.startswith("sqlite"):
        try:
            db_path = CHAINLIT_DB_URL.split("///", 1)[1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except (IndexError, OSError):
            pass

    try:
        from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

        @cl.data_layer
        def _get_data_layer() -> SQLAlchemyDataLayer:
            return SQLAlchemyDataLayer(conninfo=CHAINLIT_DB_URL)

    except Exception as _e:
        # If SQLAlchemy data layer fails to import (older chainlit, missing
        # extras), fall back to in-memory threads — UI still works, threads
        # just don't persist across restarts.
        print(f"[chainlit-ui] persistent threads disabled: {_e}")


# Only register the auth callback when persistence is enabled. Without a data
# layer, Chainlit's auth path tries to look up the user in a (nonexistent)
# database and crashes. With persistence off, skip auth entirely — Chainlit
# falls back to anonymous, in-memory sessions which is what we want for a demo.
if CHAINLIT_DB_URL:
    @cl.header_auth_callback
    def _header_auth(headers: dict) -> cl.User | None:
        # Single-user demo identity. Replace with real auth for multi-tenant.
        return cl.User(identifier="reviewer", metadata={"role": "guest"})


# ---------------------------------------------------------------------------
# Chat profiles — instant vs deep
# ---------------------------------------------------------------------------


@cl.set_chat_profiles
async def chat_profiles(_user: Any | None = None) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name="instant",
            markdown_description="**Instant** mode — fast hybrid retrieval + reranker (~3s p50).",
            icon="https://cdn.jsdelivr.net/npm/lucide-static@0.456.0/icons/zap.svg",
        ),
        cl.ChatProfile(
            name="deep",
            markdown_description=(
                "**Deep** mode — HyDE + multi-query + step-back + decomposition + "
                "two-pass verifier (~30s p50). Strongest grounding."
            ),
            icon="https://cdn.jsdelivr.net/npm/lucide-static@0.456.0/icons/microscope.svg",
        ),
    ]


# ---------------------------------------------------------------------------
# On chat start — ask for a PDF, kick off ingest, poll progress
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Thread → doc_id sidecar persistence
# Chainlit persists threads + messages via SQLAlchemyDataLayer. We additionally
# need to remember which doc_id each thread chats about, so resumed threads
# don't require a re-upload. Stored as a simple JSON keyed by thread_id.
# ---------------------------------------------------------------------------

_THREAD_DOC_PATH = Path(os.getenv("THREAD_DOC_PATH", "./data/thread_docs.json"))


def _load_thread_docs() -> dict[str, str]:
    try:
        if _THREAD_DOC_PATH.exists():
            return json.loads(_THREAD_DOC_PATH.read_text() or "{}")
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_thread_doc(thread_id: str, doc_id: str) -> None:
    if not thread_id or not doc_id:
        return
    try:
        _THREAD_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = _load_thread_docs()
        data[thread_id] = doc_id
        _THREAD_DOC_PATH.write_text(json.dumps(data))
    except OSError:
        pass


def _get_thread_id() -> str | None:
    try:
        return getattr(cl.context.session, "thread_id", None)
    except Exception:
        return None


@cl.on_chat_start
async def on_chat_start() -> None:
    profile = cl.user_session.get("chat_profile") or "instant"
    cl.user_session.set("mode", profile)
    cl.user_session.set("history", [])
    cl.user_session.set("doc_id", None)

    await cl.Message(
        content=(
            f"PDF-grounded chat. Mode: **{profile}**.\n\n"
            "I'll only answer from the PDF you upload, and I'll refuse out-of-scope queries with the fixed line:\n\n"
            f"> _{REFUSAL_STRING}_\n"
        )
    ).send()

    await _ask_for_pdf_and_ingest()


async def _ask_for_pdf_and_ingest() -> None:
    files = await cl.AskFileMessage(
        content="Upload a PDF to begin.",
        accept={"application/pdf": [".pdf"]},
        max_size_mb=200,
        timeout=600,
    ).send()
    if not files:
        await cl.Message(content="No file received. Reload the page to try again.").send()
        return

    pdf_file = files[0]
    progress_msg = cl.Message(content=f"Queued **{pdf_file.name}** for ingest…")
    await progress_msg.send()

    try:
        doc_id, summary = await _enqueue_and_poll(pdf_file.path, progress_msg)
    except Exception as e:
        await cl.ErrorMessage(content=f"Ingest failed: {e}").send()
        return

    cl.user_session.set("doc_id", doc_id)
    thread_id = _get_thread_id()
    if thread_id:
        _save_thread_doc(thread_id, doc_id)

    languages = ", ".join(summary.get("languages", []) or []) or "—"
    summary_md = (
        f"**Ingest complete.**\n\n"
        f"- doc_id: `{doc_id}`\n"
        f"- chunks: {summary.get('n_chunks', '?')}"
        f" (with tree nodes: {summary.get('n_chunks_total_with_tree', '?')})\n"
        f"- pages: {summary.get('pages', '?')}\n"
        f"- languages: {languages}\n\n"
        "Ask me anything grounded in the PDF."
    )
    progress_msg.content = summary_md
    await progress_msg.update()


@cl.on_chat_resume
async def on_chat_resume(thread: dict | None = None) -> None:
    """Restore session state when the user clicks an old thread in the sidebar."""
    profile = cl.user_session.get("chat_profile") or "instant"
    cl.user_session.set("mode", profile)
    cl.user_session.set("history", [])
    cl.user_session.set("doc_id", None)

    thread_id = (thread or {}).get("id") or _get_thread_id()
    doc_id: str | None = None
    if thread_id:
        doc_id = _load_thread_docs().get(thread_id)

    if doc_id:
        cl.user_session.set("doc_id", doc_id)
        await cl.Message(
            content=(
                f"Resumed conversation about doc `{doc_id}`. "
                f"Mode: **{profile}**. Ask away."
            )
        ).send()
    else:
        await cl.Message(
            content=(
                "Resumed an older thread, but I don't have its source PDF mapped. "
                "Upload the PDF again to continue chatting in this thread."
            )
        ).send()
        await _ask_for_pdf_and_ingest()


async def _enqueue_and_poll(pdf_path: str, progress_msg: cl.Message) -> tuple[str, dict]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            enq = (await client.post(f"{API_BASE_URL}/upload", files=files)).json()
        doc_id = enq["doc_id"]

        deadline = asyncio.get_event_loop().time() + 60 * 30
        while asyncio.get_event_loop().time() < deadline:
            sr = (await client.get(f"{API_BASE_URL}/upload/status/{doc_id}")).json()
            status = sr.get("status", "unknown")
            phase = sr.get("phase", "")
            frac = float(sr.get("fraction") or 0.0)
            bar = "▰" * int(frac * 20) + "▱" * (20 - int(frac * 20))
            progress_msg.content = (
                f"Ingesting `{doc_id}`\n\n`{bar}` {int(frac * 100)}% — **{phase}**\n\n_{sr.get('message', '')}_"
            )
            await progress_msg.update()
            if status == "done":
                return doc_id, sr.get("summary") or {}
            if status == "error":
                raise RuntimeError(sr.get("message") or "ingest failed")
            await asyncio.sleep(1.5)
    raise TimeoutError("ingest exceeded 30 minutes")


# ---------------------------------------------------------------------------
# On message — stream the answer with citation side-panel
# ---------------------------------------------------------------------------


@cl.on_message
async def on_message(message: cl.Message) -> None:
    doc_id = cl.user_session.get("doc_id")
    if not doc_id:
        await cl.ErrorMessage(content="Upload a PDF first (reload the page).").send()
        return

    mode = cl.user_session.get("mode") or "instant"
    history = cl.user_session.get("history") or []

    answer = cl.Message(content="")
    await answer.send()
    citation_elements: list[cl.Text] = []
    warnings_collected: list[str] = []
    is_refusal = False

    try:
        async for kind, payload in _stream_chat(doc_id, message.content, history, mode):
            if kind == "text":
                token = payload.get("text", "")
                answer.content += token
                await answer.stream_token(token)
            elif kind == "refusal":
                is_refusal = True
                answer.content = payload.get("text", REFUSAL_STRING)
                await answer.update()
            elif kind == "warning":
                warnings_collected.append(payload.get("text", ""))
            elif kind == "citation":
                idx = len(citation_elements) + 1
                cited = payload.get("cited_text", "")
                label = payload.get("render", f"citation {idx}")
                citation_elements.append(
                    cl.Text(name=f"[{idx}] {label}", content=cited, display="side")
                )
                # Render bbox-highlighted page images (B.6 PDF viewer)
                raw_bboxes = payload.get("bboxes") or []
                bboxes_with_pages = []
                for entry in raw_bboxes:
                    try:
                        page = int(entry["page"])
                        box = entry["bbox"]
                        bboxes_with_pages.append(
                            (page, (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                        )
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                if bboxes_with_pages:
                    try:
                        rendered = render_for_citation(doc_id, bboxes_with_pages)
                    except Exception as e:
                        rendered = []
                    for page_num, png_path in rendered:
                        citation_elements.append(
                            cl.Image(
                                name=f"[{idx}] p.{page_num}",
                                path=str(png_path),
                                display="side",
                                size="medium",
                            )
                        )
            elif kind == "error":
                await cl.ErrorMessage(content=payload.get("error", "stream error")).send()
                return
            elif kind == "done":
                break
    except Exception as e:
        await cl.ErrorMessage(content=f"Stream failed: {e}").send()
        return

    if not is_refusal and citation_elements:
        answer.elements = citation_elements
    if warnings_collected and not is_refusal:
        answer.content += "\n\n---\n" + "\n".join(warnings_collected)
    await answer.update()

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": answer.content})
    cl.user_session.set("history", history[-20:])


async def _stream_chat(
    doc_id: str, query: str, history: list[dict[str, str]], mode: str
):
    body = {"doc_id": doc_id, "query": query, "history": history, "mode": mode}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream(
            "POST",
            f"{API_BASE_URL}/chat",
            json=body,
            headers={"Accept": "text/event-stream"},
        ) as r:
            r.raise_for_status()
            event = "message"
            data_buf: list[str] = []
            async for raw in r.aiter_lines():
                line = raw.rstrip("\r")
                if line == "":
                    if data_buf:
                        try:
                            payload = json.loads("\n".join(data_buf))
                        except json.JSONDecodeError:
                            payload = {"text": "\n".join(data_buf)}
                        yield event, payload
                    event = "message"
                    data_buf = []
                elif line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_buf.append(line[len("data:"):].lstrip())
