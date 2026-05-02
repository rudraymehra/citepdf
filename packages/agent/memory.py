"""Standalone-question rewriting from chat history.

Uses Claude Haiku at temperature 0 with a strict prompt. Failure modes are
handled by falling back to the raw user message — retrieval will still work,
just not benefit from history-aware rewrites.
"""

from __future__ import annotations

import logging

import anthropic

from packages.agent.prompts import REWRITE_PROMPT
from packages.core.settings import get_settings

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


def rewrite_standalone(
    user_message: str,
    history: list[dict[str, str]],
    client: anthropic.Anthropic | None = None,
) -> str:
    """Rewrite `user_message` to a self-contained question using `history`.

    `history` is a list of {"role": "user"|"assistant", "content": str}.
    Returns the standalone question. If history is empty, returns the message.
    """
    if not history:
        return user_message.strip()

    s = get_settings()
    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)

    truncated = history[-MAX_HISTORY_TURNS:]
    history_str = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}" for turn in truncated
    )
    prompt = REWRITE_PROMPT.format(history=history_str, message=user_message)

    try:
        response = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        rewritten = "".join(text_blocks).strip()
        if not rewritten:
            return user_message.strip()
        return rewritten
    except Exception as e:
        log.warning("Standalone rewrite failed (%s); falling back to raw message", e)
        return user_message.strip()
