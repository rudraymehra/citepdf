"""Phase 1 evaluation harness.

Runs the queries in eval/sample_queries.yaml against an already-ingested PDF
and reports:
  - PASS / FAIL per query (refusal correctness for OOS; citation presence for valid)
  - Aggregate stats: refusal_precision, refusal_recall, citation_accuracy

Usage:
    python -m eval.run_eval --doc-id <doc_id> --dataset docling_arxiv

The agent.chat() function is called directly (no API needed). The doc must
already be ingested via `pdf-ingest <pdf>` so its Qdrant collection + meta
sidecar exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from eval.metrics import compute_query_metrics, render_markdown_table
from packages.agent.agent import chat
from packages.core.settings import get_settings
from packages.retrieve.retriever import retrieve as do_retrieve

console = Console()


@dataclass
class QueryResult:
    qid: str
    query: str
    is_oos_expected: bool
    answer_text: str
    citations: list[dict]
    is_refusal: bool
    passed: bool
    notes: str = ""


def run_query(doc_id: str, query: str, history: list[dict[str, str]] | None = None) -> QueryResult:
    s = get_settings()
    refusal = s.refusal_string
    text_parts: list[str] = []
    citations: list[dict] = []
    is_refusal = False

    for ev in chat(doc_id=doc_id, user_message=query, history=history or []):
        if ev.kind == "text":
            text_parts.append(ev.text or "")
        elif ev.kind == "refusal":
            is_refusal = True
            text_parts.append(ev.text or refusal)
        elif ev.kind == "citation" and ev.citation is not None:
            c = ev.citation
            citations.append(
                {
                    "chunk_id": c.chunk_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_label": c.section_label,
                    "cited_text": c.cited_text,
                }
            )
        elif ev.kind == "error":
            return QueryResult(
                qid="?", query=query, is_oos_expected=False, answer_text="",
                citations=[], is_refusal=False, passed=False, notes=f"error: {ev.error}",
            )

    answer_text = "".join(text_parts).strip()
    return QueryResult(
        qid="",
        query=query,
        is_oos_expected=False,
        answer_text=answer_text,
        citations=citations,
        is_refusal=is_refusal,
        passed=False,
    )


def evaluate_dataset(doc_id: str, dataset: dict[str, Any]) -> list[tuple[QueryResult, dict]]:
    """Returns list of (QueryResult, query_spec). query_spec preserves
    expected page ranges etc. for downstream metric computation."""
    s = get_settings()
    results: list[tuple[QueryResult, dict]] = []

    for q in dataset.get("valid_queries", []):
        r = run_query(doc_id, q["query"])
        r.qid = q["id"]
        r.is_oos_expected = False
        if r.is_refusal:
            r.passed = False
            r.notes = "refused but expected an answer"
        elif not r.citations:
            r.passed = False
            r.notes = "answered without citations"
        else:
            page_range = q.get("must_cite_pages_in_range")
            if page_range:
                ok = any(
                    page_range[0] <= c["page_start"] <= page_range[1] for c in r.citations
                )
                r.passed = ok
                r.notes = "" if ok else f"no citation in pages {page_range}"
            else:
                r.passed = True
        results.append((r, q))

    for q in dataset.get("out_of_scope_queries", []):
        r = run_query(doc_id, q["query"])
        r.qid = q["id"]
        r.is_oos_expected = True
        if r.is_refusal and r.answer_text.strip() == s.refusal_string.strip():
            r.passed = True
        else:
            r.passed = False
            r.notes = (
                f"expected refusal '{s.refusal_string}' but got: {r.answer_text[:200]}"
            )
        results.append((r, q))

    return results


def print_results(results: list[tuple[QueryResult, dict]]) -> None:
    table = Table(title="Eval Results", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type")
    table.add_column("Pass", justify="center")
    table.add_column("Citations", justify="right")
    table.add_column("Notes", max_width=60)
    table.add_column("Answer (truncated)", max_width=60)

    for r, _q in results:
        kind = "OOS" if r.is_oos_expected else "valid"
        pass_str = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        ans = r.answer_text[:80].replace("\n", " ") + ("…" if len(r.answer_text) > 80 else "")
        table.add_row(r.qid, kind, pass_str, str(len(r.citations)), r.notes, ans)
    console.print(table)

    n = len(results)
    n_pass = sum(1 for r, _q in results if r.passed)
    n_oos = sum(1 for r, _q in results if r.is_oos_expected)
    n_oos_pass = sum(1 for r, _q in results if r.is_oos_expected and r.passed)
    n_valid = n - n_oos
    n_valid_pass = n_pass - n_oos_pass
    console.print(f"\n[bold]Total[/bold]: {n_pass}/{n} passed")
    console.print(f"  Valid: {n_valid_pass}/{n_valid}")
    console.print(f"  OOS  : {n_oos_pass}/{n_oos}  (refusal recall)")


def compute_and_write_metrics(
    doc_id: str, results: list[tuple[QueryResult, dict]], output_path: Path | None
) -> str:
    """Compute RAGAS-style metrics + write a markdown table to output_path."""
    rows = []
    for r, q in results:
        # Re-retrieve the chunks the agent saw (for faithfulness scoring)
        try:
            chunks_for_metric = (
                do_retrieve(doc_id, r.query) if not r.is_refusal else None
            )
        except Exception as e:
            console.print(f"[yellow]warn[/yellow]: re-retrieve failed for {r.qid}: {e}")
            chunks_for_metric = None
        m = compute_query_metrics(
            qid=r.qid,
            is_oos_expected=r.is_oos_expected,
            question=r.query,
            answer=r.answer_text,
            citations=r.citations,
            chunks=chunks_for_metric,
            is_refusal=r.is_refusal,
            passed=r.passed,
            expected_pages_in_range=q.get("must_cite_pages_in_range"),
        )
        rows.append(m)

    md = render_markdown_table(rows)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md + "\n")
        console.print(f"[green]wrote metrics →[/green] {output_path}")
    console.print()
    console.print(md)
    return md


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True, help="doc_id of an already-ingested PDF")
    parser.add_argument("--dataset", default="docling_arxiv", help="Dataset key in sample_queries.yaml")
    parser.add_argument(
        "--queries-file",
        default=str(Path(__file__).parent / "sample_queries.yaml"),
    )
    parser.add_argument(
        "--metrics-out",
        default=str(Path(__file__).parent / "results.md"),
        help="Path to write the markdown metrics table (also printed to stdout)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip the LLM-judge metrics pass (faster, just PASS/FAIL)",
    )
    args = parser.parse_args()

    with open(args.queries_file) as f:
        cfg = yaml.safe_load(f)

    datasets = cfg.get("datasets", {})
    if args.dataset not in datasets:
        console.print(f"[red]Dataset '{args.dataset}' not in {list(datasets)}[/red]")
        sys.exit(1)

    console.print(f"[bold]Evaluating[/bold] dataset={args.dataset} on doc_id={args.doc_id}")
    results = evaluate_dataset(args.doc_id, datasets[args.dataset])
    print_results(results)

    if not args.no_metrics:
        console.print("\n[bold]Computing RAGAS-style metrics (Claude as judge)…[/bold]")
        compute_and_write_metrics(args.doc_id, results, Path(args.metrics_out))

    n_pass = sum(1 for r, _q in results if r.passed)
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
