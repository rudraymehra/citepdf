"""RAPTOR tree builder (Sarthi et al., ICLR 2024 — arXiv:2401.18059).

Build a 3-level summary tree on top of the L0 leaves:
  L0: leaves (already produced by chunker.py + contextual.py)
  L1: cluster summaries (UMAP→GMM clustering of L0 dense embeddings,
      per-cluster LLM summary)
  L2: section summaries (deterministic, one per top-level section_path)
  L3: whole-document summary (single root)

All tree nodes are stored in the same Qdrant collection as L0, distinguished
by `level`. `source_block_ids` always traces back to the L0 leaves that
informed the summary, so citations remain page-precise even when the agent
reasons from an L2 abstraction.

For very small documents (< MIN_LEAVES_FOR_RAPTOR), we skip clustering and
only build L2 + L3 — RAPTOR doesn't help when there are only a handful of
chunks, but a doc-level summary still does.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anthropic
import numpy as np
from sklearn.mixture import GaussianMixture

from packages.core.schema import Chunk
from packages.core.settings import get_settings
from packages.ingest.embedder import BGEM3Embedder

log = logging.getLogger(__name__)

# Below this leaf count, skip L1 clustering entirely. UMAP needs more samples
# than its target dimension for stable spectral init, and clustering tiny
# corpora into ~3 buckets adds noise rather than signal. L2 (per-section)
# and L3 (whole-doc) summaries still build for small docs.
MIN_LEAVES_FOR_RAPTOR = 16
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_DIM = 10
MAX_K = 20
SUMMARY_MAX_TOKENS = 400


CLUSTER_SUMMARY_PROMPT = """You summarize a cluster of related text chunks from a single document into one coherent paragraph for retrieval.

Rules:
- Capture the main themes/concepts present in the chunks.
- Be specific: use concrete entities, numbers, and technical terms that appear in the chunks.
- 200–400 words. No preamble, no markdown headings, no quotes, no lists.
- Write as a single dense paragraph, not a bulleted summary.

Chunks:
{chunks}

Summary:"""


SECTION_SUMMARY_PROMPT = """You summarize an entire document section into one coherent paragraph for retrieval.

Section title: {section}

Rules:
- Capture every important sub-topic discussed in this section.
- Use concrete entities, numbers, and technical terms from the chunks below.
- 200–400 words. No preamble, no markdown headings, no quotes.

Section chunks:
{chunks}

Summary:"""


DOC_SUMMARY_PROMPT = """You summarize an entire technical document into one coherent paragraph for retrieval.

Rules:
- Capture the document's main contributions, methods, and key findings.
- Use concrete entities, numbers, and technical terms.
- 300–500 words. No preamble, no markdown headings, no quotes.

Section summaries:
{chunks}

Document summary:"""


@dataclass
class RaptorBuild:
    chunks: list[Chunk]
    dense: np.ndarray
    sparse: list[dict]


def build_raptor_tree(
    leaves: list[Chunk],
    leaf_dense: np.ndarray,
    leaf_sparse: list[dict],
    embedder: BGEM3Embedder,
    client: anthropic.Anthropic | None = None,
) -> RaptorBuild:
    """Append L1 + L2 + L3 nodes to the leaves and return a unified bundle."""
    s = get_settings()
    if not leaves:
        return RaptorBuild(chunks=[], dense=leaf_dense, sparse=leaf_sparse)

    client = client or anthropic.Anthropic(api_key=s.anthropic_api_key)

    all_chunks: list[Chunk] = list(leaves)
    all_dense: list[np.ndarray] = [leaf_dense]
    all_sparse: list[dict] = list(leaf_sparse)

    # L1 — UMAP + GMM clustering
    l1_chunks: list[Chunk] = []
    if len(leaves) >= MIN_LEAVES_FOR_RAPTOR:
        l1_chunks = _build_l1_clusters(leaves, leaf_dense, client)
        if l1_chunks:
            l1_dense, l1_sparse = embedder.embed_texts(
                [c.text for c in l1_chunks], batch_size=8
            )
            all_chunks.extend(l1_chunks)
            all_dense.append(l1_dense)
            all_sparse.extend(l1_sparse)
            log.info("RAPTOR L1: %d cluster summaries", len(l1_chunks))
    else:
        log.info("Skipping L1 clustering — only %d leaves (< %d)", len(leaves), MIN_LEAVES_FOR_RAPTOR)

    # L2 — section summaries (deterministic by top-level section)
    l2_chunks = _build_l2_sections(leaves, client)
    if l2_chunks:
        l2_dense, l2_sparse = embedder.embed_texts([c.text for c in l2_chunks], batch_size=8)
        all_chunks.extend(l2_chunks)
        all_dense.append(l2_dense)
        all_sparse.extend(l2_sparse)
        log.info("RAPTOR L2: %d section summaries", len(l2_chunks))

    # L3 — doc summary
    l3_chunk = _build_l3_doc(leaves, l2_chunks or l1_chunks, client)
    if l3_chunk is not None:
        l3_dense, l3_sparse = embedder.embed_texts([l3_chunk.text], batch_size=1)
        all_chunks.append(l3_chunk)
        all_dense.append(l3_dense)
        all_sparse.extend(l3_sparse)
        log.info("RAPTOR L3: 1 doc summary")

    dense = np.concatenate(all_dense, axis=0) if all_dense else np.zeros((0, embedder.dim))
    return RaptorBuild(chunks=all_chunks, dense=dense, sparse=all_sparse)


def _build_l1_clusters(
    leaves: list[Chunk],
    leaf_dense: np.ndarray,
    client: anthropic.Anthropic,
) -> list[Chunk]:
    """UMAP + GMM clustering with BIC-selected k, then LLM summary per cluster."""
    import umap

    # Defense in depth — if any non-finite slipped past the embedder,
    # drop those rows from clustering instead of crashing UMAP.
    finite_mask = np.isfinite(leaf_dense).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        log.warning(
            "RAPTOR L1: %d/%d leaves had non-finite embeddings — excluding from clustering",
            n_dropped,
            len(leaf_dense),
        )
        clustering_dense = leaf_dense[finite_mask]
        clustering_leaves = [leaf for leaf, m in zip(leaves, finite_mask) if m]
    else:
        clustering_dense = leaf_dense
        clustering_leaves = leaves

    n = clustering_dense.shape[0]
    if n < MIN_LEAVES_FOR_RAPTOR:
        log.info("Skipping L1 — only %d finite leaves after sanitization", n)
        return []

    # Conservative dimensionality target: ARPACK needs comfortable headroom
    # below n. n // 3 keeps things stable down to ~16 samples; cap by UMAP_DIM.
    n_components = min(UMAP_DIM, max(2, n // 3))
    n_neighbors = min(UMAP_NEIGHBORS, max(2, n - 1))
    log.info(
        "UMAP-reducing %d leaves to %d dims (n_neighbors=%d)…",
        n,
        n_components,
        n_neighbors,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                n_components=n_components,
                metric="cosine",
                random_state=42,
            )
            reduced = reducer.fit_transform(clustering_dense)
    except Exception as e:
        log.warning(
            "UMAP failed (%s); skipping L1 cluster summaries. "
            "L2 (section) and L3 (doc) summaries will still build.",
            e,
        )
        return []

    try:
        best_k, best_gmm = _best_gmm(reduced)
    except Exception as e:
        log.warning("GMM fit failed (%s); skipping L1.", e)
        return []
    if best_gmm is None:
        return []
    labels = best_gmm.predict(reduced)
    log.info("GMM picked k=%d clusters", best_k)

    clusters: list[Chunk] = []
    for c_idx in range(best_k):
        members = [
            leaf for leaf, lbl in zip(clustering_leaves, labels) if int(lbl) == c_idx
        ]
        if len(members) < 2:
            continue
        summary = _summarize(client, CLUSTER_SUMMARY_PROMPT, members)
        if not summary.strip():
            continue
        clusters.append(_make_summary_chunk(members, summary, level=1))
    return clusters


def _best_gmm(reduced: np.ndarray) -> tuple[int, GaussianMixture | None]:
    n = reduced.shape[0]
    k_max = min(MAX_K, max(3, int(np.sqrt(n))))
    best_k, best_bic = 0, np.inf
    best_gmm: GaussianMixture | None = None
    for k in range(2, k_max + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200).fit(reduced)
        except Exception as e:
            log.debug("GMM k=%d failed: %s", k, e)
            continue
        bic = gmm.bic(reduced)
        if bic < best_bic:
            best_k, best_bic, best_gmm = k, bic, gmm
    return best_k, best_gmm


def _build_l2_sections(leaves: list[Chunk], client: anthropic.Anthropic) -> list[Chunk]:
    """Group leaves by their top-level section path; one summary per group."""
    sections: dict[str, list[Chunk]] = {}
    for leaf in leaves:
        if not leaf.section_paths:
            continue
        top = leaf.section_paths[0][0] if leaf.section_paths[0] else ""
        if not top:
            continue
        sections.setdefault(top, []).append(leaf)

    out: list[Chunk] = []
    for section_label, members in sections.items():
        if len(members) < 2:
            continue
        summary = _summarize(
            client, SECTION_SUMMARY_PROMPT, members, section=section_label
        )
        if not summary.strip():
            continue
        chunk = _make_summary_chunk(members, summary, level=2)
        chunk.section_paths = [[section_label]]
        chunk.section_anchor = _slug(section_label)
        out.append(chunk)
    return out


def _build_l3_doc(
    leaves: list[Chunk],
    upper_summaries: list[Chunk],
    client: anthropic.Anthropic,
) -> Chunk | None:
    if not leaves:
        return None
    source = upper_summaries if upper_summaries else leaves[: min(20, len(leaves))]
    summary = _summarize(client, DOC_SUMMARY_PROMPT, source)
    if not summary.strip():
        return None
    chunk = _make_summary_chunk(leaves, summary, level=3)
    chunk.section_paths = []
    chunk.section_anchor = "document"
    return chunk


def _summarize(
    client: anthropic.Anthropic,
    template: str,
    members: list[Chunk],
    **fmt: str,
) -> str:
    s = get_settings()
    snippets = "\n\n---\n\n".join(
        (m.text[:1500] for m in members[:30])  # cap context per call
    )
    prompt = template.format(chunks=snippets, **fmt)
    try:
        resp = client.messages.create(
            model=s.anthropic_generation_model,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    except Exception as e:
        log.warning("RAPTOR summarize failed: %s", e)
        return ""


def _make_summary_chunk(members: list[Chunk], summary: str, level: int) -> Chunk:
    """Members are L0 leaves (always — even L3 summary is built from a flat list of leaves)."""
    doc_id = members[0].doc_id
    pages = sorted({p for m in members for p in m.pages})
    section_paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for m in members:
        for sp in m.section_paths:
            tup = tuple(sp)
            if tup and tup not in seen:
                seen.add(tup)
                section_paths.append(sp)
    # Only collect child_ids when members are L0 (so retrieval can expand to
    # citable leaves). For L3 built from upper summaries, members are not L0
    # — _build_l3_doc passes the L0 leaves explicitly when calling this.
    leaf_child_ids = [m.chunk_id for m in members if m.level == 0]
    # Aggregate bounding boxes from leaf members so summary-node citations
    # can still highlight the underlying source regions in the PDF viewer.
    bboxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for m in members:
        if m.level == 0:
            bboxes.extend(m.block_bboxes)
    return Chunk(
        doc_id=doc_id,
        level=level,
        text=summary,
        source_block_ids=list({b for m in members for b in m.source_block_ids}),
        child_ids=leaf_child_ids,
        pages=pages,
        section_paths=section_paths,
        section_anchor=None,
        block_types=["summary"],
        language=members[0].language,
        block_bboxes=bboxes,
    )


def _slug(label: str) -> str:
    s = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
    return s[:64] or "section"
