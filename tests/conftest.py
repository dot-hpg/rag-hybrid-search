"""Shared fixtures: a tiny corpus plus fast, deterministic stand-ins for the models.

The real embedder and cross-encoder need torch and model downloads; the stand-ins
let every code path (Qdrant indexing, fusion, reranking, API) run in milliseconds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from hybrid_rag.generation.extractive import ExtractiveGenerator
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.dense import DenseRetriever, make_qdrant_client
from hybrid_rag.retrieval.tokenize import tokenize
from hybrid_rag.types import Chunk

TEXTS = {
    "lora": (
        "2106.09685",
        "LoRA",
        "LoRA freezes the pretrained weights and injects trainable rank decomposition matrices "
        "into each Transformer layer, reducing trainable parameters by 10,000 times.",
    ),
    "flash": (
        "2205.14135",
        "FlashAttention",
        "FlashAttention uses tiling to reduce memory reads and writes between GPU high bandwidth "
        "memory and on-chip SRAM. It recomputes attention in the backward pass.",
    ),
    "bert": (
        "1810.04805",
        "BERT",
        "BERT masks 15% of all WordPiece tokens and also pretrains on next sentence prediction "
        "to learn relationships between sentences.",
    ),
    "rag": (
        "2005.11401",
        "Retrieval-Augmented Generation",
        "RAG combines parametric memory from a seq2seq model with non-parametric memory, a dense "
        "vector index of Wikipedia accessed with a neural retriever.",
    ),
    "chinchilla": (
        "2203.15556",
        "Chinchilla",
        "For compute-optimal training the model size and the number of training tokens should "
        "be scaled equally. Chinchilla has 70B parameters trained on 1.4 trillion tokens.",
    ),
    "lost": (
        "2307.03172",
        "Lost in the Middle",
        "Performance is highest when relevant information occurs at the beginning or end of the "
        "input context and degrades in the middle, a U-shaped curve.",
    ),
}


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc}_{i}", doc_id=doc, title=title, page=i + 1, text=text)
        for i, (doc, title, text) in enumerate(TEXTS.values())
    ]


class HashEmbedder:
    """Bag-of-words feature hashing, L2-normalised: similar wording -> similar vectors."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class OverlapReranker:
    """Scores a passage by the fraction of query terms it contains."""

    name = "rerank"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(p))) / (len(q) or 1) for p in passages]


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def dense(chunks: list[Chunk], embedder: HashEmbedder) -> DenseRetriever:
    retriever = DenseRetriever(make_qdrant_client(), "test", embedder)
    retriever.build_index(chunks, batch_size=4)
    return retriever


@pytest.fixture
def pipeline(chunks: list[Chunk], dense: DenseRetriever) -> RAGPipeline:
    return RAGPipeline(
        chunks,
        BM25Retriever(chunks),
        dense,
        OverlapReranker(),
        ExtractiveGenerator(),
        candidate_k=10,
        rerank_depth=10,
    )
