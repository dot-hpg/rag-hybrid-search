"""Okapi BM25 over an in-memory inverted index.

Implemented directly (rather than via rank-bm25) so scoring is vectorised with numpy:
a query touches only the postings of its own terms instead of scoring every document
in Python. Building the index for the full corpus takes well under a second.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from hybrid_rag.retrieval.tokenize import tokenize
from hybrid_rag.types import Chunk, ScoredChunk


class BM25Retriever:
    name = "bm25"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer

        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        lengths = np.empty(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            tokens = tokenizer(chunk.indexed_text)
            lengths[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                docs, tfs = postings[term]
                docs.append(i)
                tfs.append(tf)

        n_docs = len(self.chunks)
        avgdl = float(lengths.mean()) or 1.0
        # Per-document length normalisation is query-independent, so precompute it.
        self._norm = k1 * (1.0 - b + b * lengths / avgdl)
        self._index: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for term, (docs, tfs) in postings.items():
            df = len(docs)
            # Lucene's non-negative IDF variant: very common terms score ~0, never < 0.
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self._index[term] = (
                np.asarray(docs, dtype=np.int32),
                np.asarray(tfs, dtype=np.float32),
                idf,
            )

    @property
    def vocabulary_size(self) -> int:
        return len(self._index)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(self.tokenizer(query)):
            entry = self._index.get(term)
            if entry is None:
                continue
            docs, tfs, idf = entry
            scores[docs] += idf * tfs * (self.k1 + 1.0) / (tfs + self._norm[docs])
        return scores

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self.scores(query)
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            ScoredChunk(self.chunks[i], float(scores[i]), {self.name: rank})
            for rank, i in enumerate(top, start=1)
            if scores[i] > 0
        ]
