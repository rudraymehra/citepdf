"""Faithfulness gate using Claude Haiku as the LLM-judge.

Phase 1 substitute for bespoke-minicheck-7B. Same goal: detect sentences in
the answer that aren't supported by the retrieved chunks, and either edit
them out or trigger a refusal.

Phase 3 will swap this for actual bespoke-minicheck NLI inference.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

from packages.agent.prompts import JUDGE_PROMPT
from packages.core.settings import get_settings
from packages.retrieve.retriever import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    score: float
    supported: bool
    unsupported_claims: list[str]
    raw: dict | None = None


def check_faithfulness(
    answer: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> FaithfulnessResult:
    s = get_settings()
    if not answer.strip() or answer.strip() == s.refusal_string:
        # Refusals trivially pass (nothing to ground)
        return FaithfulnessResult(score=1.0, supported=True, unsupported_claims=[])

    if not chunks:
        return FaithfulnessResult(score=0.0, supported=False, unsupported_claims=[answer])

    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)
    chunks_str = "\n\n".join(
        f"[chunk {i}] page {c.page_start}, {c.section_label}:\n{c.text[:1500]}"
        for i, c in enumerate(chunks[:8])
    )
    prompt = JUDGE_PROMPT.format(answer=answer, chunks=chunks_str)

    try:
        response = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=1500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        parsed = _parse_judge_json(text)
    except Exception as e:
        log.warning("Faithfulness LLM-judge failed: %s", e)
        # Fail open on judge errors — let the user see the answer rather than refusing.
        return FaithfulnessResult(score=1.0, supported=True, unsupported_claims=[], raw={"error": str(e)})

    if parsed is None:
        log.warning("Faithfulness LLM-judge returned unparsable JSON; failing open")
        return FaithfulnessResult(score=1.0, supported=True, unsupported_claims=[], raw={"text": text})

    score = float(parsed.get("score", 0.0))
    unsupported = [c for c in parsed.get("unsupported", []) if isinstance(c, str)]
    return FaithfulnessResult(
        score=score,
        supported=score >= s.faithfulness_threshold,
        unsupported_claims=unsupported,
        raw=parsed,
    )


def _parse_judge_json(text: str) -> dict | None:
    """Extract a JSON object from the judge's response.

    The judge prompt asks for raw JSON, but models occasionally add fences
    or preamble. Strip them and parse."""
    text = text.strip()
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
