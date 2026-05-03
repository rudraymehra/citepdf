"""Streamlit demo frontend.

Run with:
    streamlit run apps/frontend/streamlit_app.py

Sidebar: file upload (calls API /upload, shows ingest summary).
Main: chat box; SSE-streamed answers from API /chat with inline citations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REFUSAL_STRING = os.getenv("REFUSAL_STRING", "I cannot answer this from the provided document.")

st.set_page_config(
    page_title="PDF Chat — Strict Grounding",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    st.session_state.setdefault("doc_id", None)
    st.session_state.setdefault("doc_meta", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("citations_by_idx", {})
    st.session_state.setdefault("mode", "instant")


def _upload_pdf(file) -> dict[str, Any]:
    """Enqueue async ingest and poll status until done."""
    import time as _time
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
        r = client.post(f"{API_BASE_URL}/upload", files=files)
        r.raise_for_status()
        enq = r.json()
        doc_id = enq["doc_id"]

        progress_bar = st.sidebar.progress(0.0, text="Queued…")
        deadline = _time.time() + 60 * 30  # 30 min hard cap
        while _time.time() < deadline:
            sr = client.get(f"{API_BASE_URL}/upload/status/{doc_id}").json()
            status = sr.get("status", "unknown")
            phase = sr.get("phase", "")
            frac = float(sr.get("fraction") or 0.0)
            msg = sr.get("message", "") or phase
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=f"{phase}: {msg}")
            if status == "done":
                summary = sr.get("summary") or {}
                return {
                    "doc_id": doc_id,
                    "n_blocks": int(summary.get("n_blocks", 0)),
                    "n_chunks": int(summary.get("n_chunks", 0)),
                    "pages": int(summary.get("pages", 0)),
                    "languages": list(summary.get("languages", []) or []),
                }
            if status == "error":
                raise RuntimeError(sr.get("message") or "ingest failed")
            _time.sleep(2.0)
    raise TimeoutError("ingest exceeded 30 minutes")


def _stream_chat(
    doc_id: str, query: str, history: list[dict[str, str]], mode: str
) -> Iterator[tuple[str, dict]]:
    """Yield (event_kind, payload) tuples from the SSE endpoint."""
    body = {"doc_id": doc_id, "query": query, "history": history, "mode": mode}
    headers = {"Accept": "text/event-stream"}
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        with client.stream("POST", f"{API_BASE_URL}/chat", json=body, headers=headers) as r:
            r.raise_for_status()
            event = "message"
            data_buf: list[str] = []
            for raw in r.iter_lines():
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
                # ignore other SSE fields (id:, retry:, comments)


def _render_history() -> None:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("citations"):
                with st.expander(f"Citations ({len(turn['citations'])})", expanded=False):
                    for c in turn["citations"]:
                        st.markdown(
                            f"- **{c.get('render', '')}** — _{c.get('cited_text', '')}_"
                        )


def _sidebar() -> None:
    st.sidebar.title("PDF Chat — Task 3")
    st.sidebar.caption("Strict grounding · citations on every claim · refuse OOS")

    file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if file is not None and st.sidebar.button("Ingest", type="primary"):
        with st.sidebar.status("Ingesting — first run downloads BGE-M3 (~1.2GB)…", expanded=True) as s:
            try:
                meta = _upload_pdf(file)
                st.session_state.doc_id = meta["doc_id"]
                st.session_state.doc_meta = meta
                st.session_state.history = []
                s.update(label="Ingest complete ✓", state="complete")
            except Exception as e:
                s.update(label=f"Ingest failed: {e}", state="error")

    if st.session_state.doc_meta:
        m = st.session_state.doc_meta
        st.sidebar.success(f"doc_id: `{m['doc_id'][:8]}…`")
        st.sidebar.metric("Chunks", m["n_chunks"])
        st.sidebar.metric("Pages", m["pages"])
        st.sidebar.write("Languages:", ", ".join(m["languages"]) or "—")

    st.sidebar.divider()
    st.session_state.mode = st.sidebar.radio(
        "Mode",
        options=["instant", "deep"],
        index=0,
        help="Phase 1 only implements instant. Deep is wired in Phase 2.",
    )
    if st.sidebar.button("Reset chat"):
        st.session_state.history = []
        st.rerun()


def _handle_user_query(query: str) -> None:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        cite_box = st.empty()
        accumulated_text = ""
        citations: list[dict] = []
        is_refusal = False

        warnings_text: list[str] = []
        try:
            stream = _stream_chat(
                doc_id=st.session_state.doc_id,
                query=query,
                history=[
                    {"role": h["role"], "content": h["content"]}
                    for h in st.session_state.history[:-1]
                ],
                mode=st.session_state.mode,
            )
            for kind, payload in stream:
                if kind == "text":
                    accumulated_text += payload.get("text", "")
                    placeholder.markdown(accumulated_text)
                elif kind == "refusal":
                    accumulated_text = payload.get("text", REFUSAL_STRING)
                    is_refusal = True
                    placeholder.markdown(f"**:red[{accumulated_text}]**")
                elif kind == "warning":
                    warnings_text.append(payload.get("text", ""))
                elif kind == "citation":
                    citations.append(payload)
                elif kind == "error":
                    placeholder.error(payload.get("error", "error"))
                elif kind == "done":
                    break
        except Exception as e:
            placeholder.error(f"Streaming failed: {e}")
            accumulated_text = f"[error: {e}]"

        if warnings_text and not is_refusal:
            placeholder.markdown(
                accumulated_text + "\n\n---\n" + "\n".join(warnings_text)
            )

        if citations and not is_refusal:
            with cite_box.expander(f"Citations ({len(citations)})", expanded=False):
                for c in citations:
                    st.markdown(f"- **{c.get('render', '')}** — _{c.get('cited_text', '')}_")

    st.session_state.history.append(
        {"role": "assistant", "content": accumulated_text, "citations": citations}
    )


def main() -> None:
    _init_state()
    _sidebar()

    if not st.session_state.doc_id:
        st.title("Upload a PDF to start chatting")
        st.write(
            "Upload a PDF in the sidebar. The agent will only answer questions grounded in that PDF "
            "and will refuse out-of-scope queries with a fixed message."
        )
        st.write(
            f"**Refusal string:** `{REFUSAL_STRING}`"
        )
        return

    st.title("Chat")
    _render_history()

    user_query = st.chat_input("Ask a question about the PDF…")
    if user_query:
        _handle_user_query(user_query)
        st.rerun()


if __name__ == "__main__":
    main()
