"""The retrieval pipeline: first-stage retrievers -> fusion -> reranking -> generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from hybrid_rag.config import Settings
from hybrid_rag.corpus import load_chunks
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.fusion import min_max_fusion, reciprocal_rank_fusion
from hybrid_rag.retrieval.rerank import Reranker, rerank
from hybrid_rag.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from hybrid_rag.generation.base import Answer, Generator
    from hybrid_rag.retrieval.dense import DenseRetriever

log = logging.getLogger(__name__)


class Mode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    DENSE_RERANK = "dense_rerank"
    HYBRID_RERANK = "hybrid_rerank"

    @property
    def needs_dense(self) -> bool:
        return self is not Mode.BM25

    @property
    def needs_reranker(self) -> bool:
        return self in (Mode.DENSE_RERANK, Mode.HYBRID_RERANK)


@dataclass
class SearchResult:
    query: str
    mode: Mode
    hits: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


class _Timer:
    def __init__(self, timings: dict[str, float], stage: str) -> None:
        self.timings, self.stage = timings, stage

    def __enter__(self) -> None:
        self.t0 = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self.timings[self.stage] = round((time.perf_counter() - self.t0) * 1000, 2)


class RAGPipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Retriever,
        dense: DenseRetriever | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        rerank_depth: int = 30,
        fusion: str = "rrf",
    ) -> None:
        if fusion not in ("rrf", "minmax"):
            raise ValueError("fusion must be 'rrf' or 'minmax'")
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.generator = generator
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.fusion = fusion
        self.rerank_depth = rerank_depth

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        load_dense: bool = True,
        load_reranker: bool = True,
        load_generator: bool = True,
    ) -> RAGPipeline:
        chunks = load_chunks(settings.chunks_file)
        bm25 = BM25Retriever(chunks)
        log.info("BM25 index: %d chunks, %d terms", len(chunks), bm25.vocabulary_size)

        dense = None
        if load_dense:
            from hybrid_rag.retrieval.dense import (
                DenseRetriever,
                SentenceTransformerEmbedder,
                make_qdrant_client,
            )

            embedder = SentenceTransformerEmbedder(
                settings.embedding_model, settings.query_instruction, settings.device
            )
            client = make_qdrant_client(
                settings.qdrant_url, str(settings.qdrant_path), settings.qdrant_api_key
            )
            dense = DenseRetriever(client, settings.collection, embedder)
            if not dense.is_indexed(expected_count=len(chunks)):
                if not settings.build_index_on_startup:
                    raise RuntimeError("Vector index missing or stale. Run `rag index`.")
                log.info("Vector index missing or stale; building it now")
                dense.build_index(chunks)

        reranker = None
        if load_reranker:
            from hybrid_rag.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings.reranker_model, settings.device)

        generator = None
        if load_generator:
            from hybrid_rag.generation import make_generator

            generator = make_generator(settings)

        return cls(
            chunks,
            bm25,
            dense,
            reranker,
            generator,
            candidate_k=settings.candidate_k,
            rrf_k=settings.rrf_k,
            fusion=settings.fusion,
            rerank_depth=settings.rerank_depth,
        )

    @property
    def available_modes(self) -> list[Mode]:
        return [
            m
            for m in Mode
            if (self.dense or not m.needs_dense) and (self.reranker or not m.needs_reranker)
        ]

    def search(self, query: str, mode: Mode | str = Mode.HYBRID_RERANK, k: int = 5) -> SearchResult:
        mode = Mode(mode)
        if mode not in self.available_modes:
            raise ValueError(f"mode {mode.value!r} is not available in this deployment")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        timings: dict[str, float] = {}
        depth = max(k, self.candidate_k)

        if mode is Mode.BM25:
            with _Timer(timings, "bm25"):
                hits = self.bm25.search(query, k)
            return SearchResult(query, mode, hits, timings)

        assert self.dense is not None
        with _Timer(timings, "dense"):
            dense_hits = self.dense.search(query, depth)

        if mode in (Mode.DENSE, Mode.DENSE_RERANK):
            candidates = dense_hits
        else:
            with _Timer(timings, "bm25"):
                bm25_hits = self.bm25.search(query, depth)
            with _Timer(timings, "fusion"):
                if self.fusion == "rrf":
                    candidates = reciprocal_rank_fusion([bm25_hits, dense_hits], k=self.rrf_k)
                else:
                    candidates = min_max_fusion([bm25_hits, dense_hits])
                for rank, hit in enumerate(candidates, start=1):
                    hit.ranks["fused"] = rank

        if mode.needs_reranker:
            assert self.reranker is not None
            with _Timer(timings, "rerank"):
                hits = rerank(self.reranker, query, candidates[: self.rerank_depth], k)
        else:
            hits = candidates[:k]
        return SearchResult(query, mode, hits, timings)

    def _require_generator(self) -> Generator:
        if self.generator is None:
            raise RuntimeError("no answer generator configured")
        return self.generator

    def ask(
        self, query: str, mode: Mode | str = Mode.HYBRID_RERANK, k: int = 5
    ) -> tuple[SearchResult, Answer]:
        result = self.search(query, mode, k)
        generator = self._require_generator()
        t0 = time.perf_counter()
        answer = generator.generate(result.query, result.hits)
        result.timings_ms["generate"] = round((time.perf_counter() - t0) * 1000, 2)
        return result, answer

    def ask_stream(
        self, query: str, mode: Mode | str = Mode.HYBRID_RERANK, k: int = 5
    ) -> tuple[SearchResult, Iterator[str]]:
        result = self.search(query, mode, k)
        return result, self._require_generator().stream(result.query, result.hits)
