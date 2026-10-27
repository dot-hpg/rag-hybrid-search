"""Dense retrieval: sentence-transformer embeddings stored and searched in Qdrant."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from hybrid_rag.types import Chunk, ScoredChunk

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers bi-encoder. Embeddings are L2-normalised so
    cosine similarity equals the dot product."""

    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            self.query_instruction + text, normalize_embeddings=True, convert_to_numpy=True
        )


def make_qdrant_client(
    url: str | None = None, path: str | None = None, api_key: str | None = None
) -> QdrantClient:
    """A Qdrant server client when ``url`` is given, otherwise embedded on-disk mode
    (``path``) or in-memory mode (neither)."""
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if path:
        return QdrantClient(path=path)
    return QdrantClient(location=":memory:")


class DenseRetriever:
    name = "dense"

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def is_indexed(self, expected_count: int | None = None) -> bool:
        if not self.client.collection_exists(self.collection):
            return False
        if expected_count is None:
            return True
        return self.client.count(self.collection).count == expected_count

    def build_index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        """(Re)create the collection and embed + upsert every chunk."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding", unit="batch"):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_documents([c.indexed_text for c in batch])
            self.client.upsert(
                self.collection,
                points=[
                    PointStruct(id=start + i, vector=vec.tolist(), payload=chunk.to_dict())
                    for i, (chunk, vec) in enumerate(zip(batch, vectors, strict=True))
                ],
            )
        count = self.client.count(self.collection).count
        log.info("indexed %d chunks into %s", count, self.collection)
        return count

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        vector = self.embedder.embed_query(query)
        result = self.client.query_points(
            self.collection, query=vector.tolist(), limit=k, with_payload=True
        )
        return [
            ScoredChunk(Chunk.from_dict(p.payload or {}), float(p.score), {self.name: rank})
            for rank, p in enumerate(result.points, start=1)
        ]
