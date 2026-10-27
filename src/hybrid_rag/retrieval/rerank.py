"""Second-stage reranking with a cross-encoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hybrid_rag.types import ScoredChunk


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly with a cross-encoder.

    Bi-encoders embed query and passage separately, so they can't model token-level
    interactions between the two; a cross-encoder reads them together and is far more
    precise, but costs a forward pass per pair. That's why it only reorders the small
    fused candidate set instead of searching the whole corpus.
    """

    name = "rerank"

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, keep lazy

        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        scores = self.model.predict(
            [(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


def rerank(
    reranker: Reranker, query: str, candidates: Sequence[ScoredChunk], k: int
) -> list[ScoredChunk]:
    """Re-score ``candidates`` with ``reranker`` and return the top ``k``."""
    if not candidates:
        return []
    scores = reranker.score(query, [c.chunk.indexed_text for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        ScoredChunk(candidates[i].chunk, scores[i], {**candidates[i].ranks, reranker.name: rank})
        for rank, i in enumerate(order, start=1)
    ]
