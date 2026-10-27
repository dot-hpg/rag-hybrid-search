"""Integration tests against the real embedding and reranking models.

Skipped unless sentence-transformers is installed; the benchmark workflow runs them.
"""

import pytest

pytest.importorskip("sentence_transformers")

from hybrid_rag.config import Settings  # noqa: E402
from hybrid_rag.retrieval.dense import SentenceTransformerEmbedder  # noqa: E402
from hybrid_rag.retrieval.rerank import CrossEncoderReranker  # noqa: E402

pytestmark = pytest.mark.slow


def test_embedder_is_normalised_and_query_aware():
    s = Settings()
    emb = SentenceTransformerEmbedder(s.embedding_model, s.query_instruction)
    docs = emb.embed_documents(
        ["LoRA injects trainable low-rank matrices.", "FlashAttention tiles attention in SRAM."]
    )
    q = emb.embed_query("How does LoRA reduce trainable parameters?")
    assert docs.shape == (2, emb.dim)
    assert abs(float((q**2).sum()) - 1.0) < 1e-4
    assert float(docs[0] @ q) > float(docs[1] @ q)


def test_cross_encoder_prefers_the_answering_passage():
    reranker = CrossEncoderReranker(Settings().reranker_model)
    scores = reranker.score(
        "What percentage of tokens does BERT mask?",
        [
            "BERT masks 15% of all WordPiece tokens in each sequence at random.",
            "Chinchilla was trained on 1.4 trillion tokens.",
        ],
    )
    assert scores[0] > scores[1]
