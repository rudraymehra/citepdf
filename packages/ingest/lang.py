"""Per-chunk language detection using fast-langdetect (cld3 wrapper).

Returns ISO 639-1 codes. For mixed-script chunks we just pick the dominant
language; cross-script handling is Phase 3 polish.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def detect_lang(text: str) -> str:
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        from fast_langdetect import detect

        flat = text.replace("\n", " ").strip()[:2000]
        result = detect(flat, low_memory=True)
        lang = result.get("lang") if isinstance(result, dict) else None
        return lang or "en"
    except Exception as e:
        log.debug("lang detect failed: %s", e)
        return "en"


def dominant_language(langs: list[str]) -> str:
    """Pick the most common language tag from a list."""
    if not langs:
        return "en"
    counts: dict[str, int] = {}
    for lg in langs:
        counts[lg] = counts.get(lg, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]
