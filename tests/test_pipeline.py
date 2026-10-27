import pytest

from hybrid_rag.pipeline import Mode, RAGPipeline
from hybrid_rag.retrieval.bm25 import BM25Retriever


def test_dense_index_round_trip(dense, chunks):
    assert dense.is_indexed(expected_count=len(chunks))
    assert not dense.is_indexed(expected_count=len(chunks) + 1)
    hits = dense.search("rank decomposition matrices trainable", k=2)
    assert hits[0].chunk.doc_id == "2106.09685"
    assert hits[0].chunk == chunks[0]  # payload round-trips through Qdrant
    assert hits[0].ranks == {"dense": 1}


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_finds_the_obvious_answer(pipeline, mode):
    result = pipeline.search("How does FlashAttention use tiling with SRAM?", mode, k=3)
    assert result.hits[0].chunk.doc_id == "2205.14135"
    assert len(result.hits) <= 3
    assert result.total_ms >= 0


def test_stage_timings_and_rank_provenance(pipeline):
    result = pipeline.search("training tokens compute-optimal", Mode.HYBRID_RERANK, k=2)
    assert set(result.timings_ms) == {"dense", "bm25", "fusion", "rerank"}
    top = result.hits[0]
    assert {"fused", "rerank"} <= set(top.ranks)
    assert top.ranks.keys() & {"bm25", "dense"}


def test_rejects_empty_query_and_unavailable_mode(chunks):
    bm25_only = RAGPipeline(chunks, BM25Retriever(chunks))
    assert bm25_only.available_modes == [Mode.BM25]
    with pytest.raises(ValueError, match="not available"):
        bm25_only.search("anything", Mode.HYBRID)
    with pytest.raises(ValueError, match="empty"):
        bm25_only.search("   ", Mode.BM25)
    with pytest.raises(RuntimeError):
        bm25_only.ask("anything", Mode.BM25)


def test_ask_returns_cited_answer(pipeline):
    result, answer = pipeline.ask("What percentage of WordPiece tokens does BERT mask?", k=3)
    assert "15%" in answer.text
    assert answer.citations and all(1 <= c <= len(result.hits) for c in answer.citations)
    assert "generate" in result.timings_ms
