"""Two-pass verifier — corrects the generated answer by dropping unsupported claims.

Architecture doc §6.4: a separate Claude Sonnet call is asked to identify
unsupported claims; we then drop those sentences from the answer. If the
answer becomes empty, the caller refuses with the fixed refusal string.

Used in deep mode AFTER generation, BEFORE the faithfulness gate.

Also exposes find_unsupported_claims() so the CRAG retry loop can re-retrieve
on those claims rather than just dropping them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

from packages.core.settings import get_settings
from packages.retrieve.retriever import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Structured verifier output used by both the warning path and CRAG retry."""

    corrected_text: str  # answer with unsupported sentences dropped
    unsupported: list[str]  # the dropped sentences themselves


VERIFIER_PROMPT = """You are a strict citation auditor. You will receive an ANSWER and the SOURCE CHUNKS that were retrieved from a single PDF.

Task: identify every sentence in the ANSWER that is NOT directly supported by the SOURCE CHUNKS. A sentence is supported only if the source chunks contain the same claim (paraphrase OK; pure inference NOT OK).

Output a JSON object with:
  "unsupported_sentences": [list of exact substrings of the answer that should be dropped]

If everything is supported, output {{"unsupported_sentences": []}}. Output ONLY the JSON, no preamble.

ANSWER:
{answer}

SOURCE CHUNKS:
{chunks}

JSON:"""


def run_verifier(
    answer: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> VerifyResult:
    """Run the verifier and return a structured result (corrected text + dropped claims)."""
    s = get_settings()
    if not answer.strip():
        return VerifyResult(corrected_text=answer, unsupported=[])

    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)
    chunks_str = "\n\n".join(
        f"[chunk {i}] page {c.page_start}, {c.section_label}:\n{c.text[:1500]}"
        for i, c in enumerate(chunks[:8])
    )
    prompt = VERIFIER_PROMPT.format(answer=answer, chunks=chunks_str)

    try:
        resp = client.messages.create(
            model=s.anthropic_generation_model,
            max_tokens=1500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        parsed = _parse_json(text)
    except Exception as e:
        log.warning("verifier call failed (%s); treating as fully supported", e)
        return VerifyResult(corrected_text=answer, unsupported=[])

    if not parsed:
        return VerifyResult(corrected_text=answer, unsupported=[])

    raw_unsupported = parsed.get("unsupported_sentences", [])
    unsupported = [b.strip() for b in raw_unsupported if isinstance(b, str) and b.strip()]
    if not unsupported:
        return VerifyResult(corrected_text=answer, unsupported=[])

    corrected = answer
    for bad in unsupported:
        corrected = corrected.replace(bad, "")
    corrected = re.sub(r"\s+", " ", corrected).strip()
    log.info("verifier flagged %d unsupported claim(s)", len(unsupported))
    return VerifyResult(corrected_text=corrected, unsupported=unsupported)


def verify_and_correct(
    answer: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> str:
    """Backward-compat alias — returns just the corrected text."""
    return run_verifier(answer, chunks, client=client).corrected_text


def find_unsupported_claims(
    answer: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> list[str]:
    """Return only the unsupported claims (used by the CRAG retry loop)."""
    return run_verifier(answer, chunks, client=client).unsupported


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
