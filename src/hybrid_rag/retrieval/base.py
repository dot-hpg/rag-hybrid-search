"""Retriever interface."""

from __future__ import annotations

from typing import Protocol

from hybrid_rag.types import ScoredChunk


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` chunks, best first, each with ``ranks[self.name]`` set."""
        ...
