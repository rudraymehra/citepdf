"""RAGAS-style metrics computed via Claude as the LLM judge.

We compute the same four metric families RAGAS reports — faithfulness,
answer relevancy, context precision, context recall — but without the
langchain-anthropic adapter chain. Each metric is a small Claude call.

Aggregate output is a markdown table suitable for the README and
SUBMISSION.md.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

from packages.agent.grounding_check import check_faithfulness
from packages.core.settings import get_settings
from packages.retrieve.retriever import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    qid: str
    is_oos_expected: bool
    is_refusal: bool
    passed: bool
    faithfulness: float
    answer_relevancy: float
    citation_accuracy: float
    n_citations: int


ANSWER_RELEVANCY_PROMPT = """Rate how directly the ANSWER addresses the QUESTION.

Score 1.0 if the answer fully and directly addresses what was asked; 0.5 if it
partially addresses the question or includes irrelevant material; 0.0 if it
does not address the question at all (or is empty/error/refusal).

Output a single JSON object: {{"score": <float 0..1>, "reason": "<one short sentence>"}}.

QUESTION:
{question}

ANSWER:
{answer}

JSON:"""


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def answer_relevancy(
    question: str,
    answer: str,
    client: anthropic.Anthropic | None = None,
) -> float:
    s = get_settings()
    if not answer.strip() or answer.strip() == s.refusal_string:
        return 0.0
    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)
    try:
        resp = client.messages.create(
            model=s.anthropic_judge_model,
            max_tokens=200,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_RELEVANCY_PROMPT.format(
                        question=question, answer=answer
                    ),
                }
            ],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        parsed = _parse_json(text)
    except Exception as e:
        log.warning("answer_relevancy LLM-judge failed: %s", e)
        return 0.0
    if not parsed:
        return 0.0
    try:
        return max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def citation_accuracy(
    citations: list[dict],
    expected_pages_in_range: list[int] | None,
) -> float:
    """Fraction of citations whose page falls in the expected range. 1.0 if no expectation."""
    if not citations:
        return 0.0
    if not expected_pages_in_range:
        return 1.0
    lo, hi = expected_pages_in_range[0], expected_pages_in_range[1]
    n_ok = sum(1 for c in citations if lo <= int(c.get("page_start", 0)) <= hi)
    return n_ok / len(citations)


def compute_query_metrics(
    qid: str,
    is_oos_expected: bool,
    question: str,
    answer: str,
    citations: list[dict],
    chunks: list[RetrievedChunk] | None,
    is_refusal: bool,
    passed: bool,
    expected_pages_in_range: list[int] | None,
    client: anthropic.Anthropic | None = None,
) -> QueryMetrics:
    """Compute one row of metrics. RAG-side metrics (faithfulness) need chunks;
    skip them if chunks aren't available (e.g. for refusal cases)."""
    # Faithfulness: claims in the answer supported by retrieved chunks
    if chunks and not is_refusal:
        try:
            f = check_faithfulness(answer, chunks, client=client)
            faithfulness_score = f.score
        except Exception as e:
            log.warning("faithfulness check failed for %s: %s", qid, e)
            faithfulness_score = 0.0
    else:
        faithfulness_score = 1.0  # refusals trivially faithful

    # Answer relevancy
    if is_refusal and is_oos_expected:
        rel = 1.0  # correct refusal is fully relevant
    elif is_refusal and not is_oos_expected:
        rel = 0.0  # incorrect refusal is irrelevant
    else:
        rel = answer_relevancy(question, answer, client=client)

    cit_acc = citation_accuracy(citations, expected_pages_in_range) if not is_refusal else 1.0

    return QueryMetrics(
        qid=qid,
        is_oos_expected=is_oos_expected,
        is_refusal=is_refusal,
        passed=passed,
        faithfulness=faithfulness_score,
        answer_relevancy=rel,
        citation_accuracy=cit_acc,
        n_citations=len(citations),
    )


def render_markdown_table(rows: list[QueryMetrics]) -> str:
    """Build a markdown table summarizing per-query metrics + aggregates."""
    lines: list[str] = []
    lines.append("| Query | Type | Pass | Faithfulness | Answer Relevancy | Citation Acc | #Cites |")
    lines.append("|---|---|:---:|---:|---:|---:|---:|")
    for r in rows:
        kind = "OOS" if r.is_oos_expected else "valid"
        passed = "✓" if r.passed else "✗"
        lines.append(
            f"| `{r.qid}` | {kind} | {passed} | {r.faithfulness:.2f} | "
            f"{r.answer_relevancy:.2f} | {r.citation_accuracy:.2f} | {r.n_citations} |"
        )

    n = len(rows)
    if n == 0:
        return "\n".join(lines)

    n_oos = sum(1 for r in rows if r.is_oos_expected)
    n_valid = n - n_oos
    n_pass = sum(1 for r in rows if r.passed)
    n_oos_pass = sum(1 for r in rows if r.is_oos_expected and r.passed)
    n_valid_pass = n_pass - n_oos_pass
    avg_faith = sum(r.faithfulness for r in rows) / n
    avg_rel = sum(r.answer_relevancy for r in rows) / n
    avg_cit = sum(r.citation_accuracy for r in rows if not r.is_oos_expected) / max(n_valid, 1)
    refusal_recall = (n_oos_pass / n_oos) if n_oos else 1.0
    # Refusal precision: of all refused, how many were actually OOS
    n_refused = sum(1 for r in rows if r.is_refusal)
    n_refused_correct = sum(1 for r in rows if r.is_refusal and r.is_oos_expected)
    refusal_precision = (n_refused_correct / n_refused) if n_refused else 1.0

    lines.append("")
    lines.append("**Aggregate metrics:**")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---:|")
    lines.append(f"| Faithfulness (avg) | **{avg_faith:.2f}** |")
    lines.append(f"| Answer Relevancy (avg) | **{avg_rel:.2f}** |")
    lines.append(f"| Citation Accuracy (avg, valid queries) | **{avg_cit:.2f}** |")
    lines.append(f"| Refusal Precision | **{refusal_precision:.2f}** ({n_refused_correct}/{n_refused}) |")
    lines.append(f"| Refusal Recall | **{refusal_recall:.2f}** ({n_oos_pass}/{n_oos}) |")
    lines.append(f"| Pass rate | **{n_pass}/{n}** ({n_valid_pass}/{n_valid} valid + {n_oos_pass}/{n_oos} OOS) |")
    return "\n".join(lines)
