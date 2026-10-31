"""Core data types shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    page: int
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(d["chunk_id"]),
            doc_id=str(d["doc_id"]),
            title=str(d["title"]),
            page=int(d["page"]),
            text=str(d["text"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def indexed_text(self) -> str:
        """Text fed to the retrievers. Prefixing the paper title gives every chunk
        document-level context ("contextual chunk header"), which helps both BM25
        and the embedder on chunks that never mention the paper's subject by name."""
        return f"{self.title}. {self.text}"


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    # 1-based rank of this chunk in each stage that saw it, e.g. {"bm25": 3, "dense": 1}.
    # Surfaced in the API/UI so you can see *why* a result ended up where it did.
    ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**self.chunk.to_dict(), "score": round(float(self.score), 6), "ranks": self.ranks}
