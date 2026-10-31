"""Combine ranked lists from several retrievers into one."""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_rag.types import ScoredChunk


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    ``score(d) = sum_i w_i / (k + rank_i(d))`` over the lists that contain ``d``.

    RRF only looks at ranks, so it needs no score calibration between retrievers
    whose scores live on different scales (BM25 is unbounded, cosine is in [-1, 1]).
    ``k`` damps the advantage of the very top ranks; 60 is the value from the paper.
    The per-retriever ranks are merged into each fused result for explainability.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("need one weight per result list")

    fused: dict[str, ScoredChunk] = {}
    for results, weight in zip(result_lists, weights, strict=True):
        for rank, hit in enumerate(results, start=1):
            entry = fused.get(hit.chunk.chunk_id)
            if entry is None:
                entry = fused[hit.chunk.chunk_id] = ScoredChunk(hit.chunk, 0.0, {})
            entry.score += weight / (k + rank)
            entry.ranks.update(hit.ranks)

    ordered = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    return ordered[:limit] if limit is not None else ordered


def min_max_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Convex combination of per-list min-max normalised scores.

    Included as the classic alternative to RRF for the benchmark's ablation. It is
    sensitive to score outliers and to how deep each list is, which RRF avoids.
    """
    if weights is None:
        weights = [1.0 / len(result_lists)] * len(result_lists)
    fused: dict[str, ScoredChunk] = {}
    for results, weight in zip(result_lists, weights, strict=True):
        if not results:
            continue
        scores = [h.score for h in results]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for hit in results:
            entry = fused.get(hit.chunk.chunk_id)
            if entry is None:
                entry = fused[hit.chunk.chunk_id] = ScoredChunk(hit.chunk, 0.0, {})
            entry.score += weight * (hit.score - lo) / span
            entry.ranks.update(hit.ranks)
    ordered = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    return ordered[:limit] if limit is not None else ordered
