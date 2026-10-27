import pytest

from hybrid_rag.retrieval.fusion import min_max_fusion, reciprocal_rank_fusion
from hybrid_rag.retrieval.rerank import rerank
from hybrid_rag.types import Chunk, ScoredChunk


def _hits(name: str, ids: list[str], scores: list[float] | None = None) -> list[ScoredChunk]:
    scores = scores or [1.0 / (i + 1) for i in range(len(ids))]
    return [
        ScoredChunk(Chunk(cid, "d", "T", 1, f"text {cid}"), s, {name: r})
        for r, (cid, s) in enumerate(zip(ids, scores, strict=True), start=1)
    ]


def test_rrf_scores_and_merged_ranks():
    fused = reciprocal_rank_fusion(
        [_hits("bm25", ["a", "b", "c"]), _hits("dense", ["b", "d"])], k=60
    )
    by_id = {h.chunk.chunk_id: h for h in fused}
    assert fused[0].chunk.chunk_id == "b"  # in both lists
    assert by_id["b"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["a"].score == pytest.approx(1 / 61)
    assert by_id["b"].ranks == {"bm25": 2, "dense": 1}
    assert by_id["d"].ranks == {"dense": 2}


def test_rrf_weights_and_limit():
    lists = [_hits("bm25", ["a"]), _hits("dense", ["b"])]
    fused = reciprocal_rank_fusion(lists, weights=[1.0, 2.0], limit=1)
    assert [h.chunk.chunk_id for h in fused] == ["b"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(lists, weights=[1.0])


def test_rrf_does_not_mutate_inputs():
    bm25 = _hits("bm25", ["a"])
    reciprocal_rank_fusion([bm25, _hits("dense", ["a"])])
    assert bm25[0].ranks == {"bm25": 1}


def test_min_max_fusion_normalises_scales():
    bm25 = _hits("bm25", ["a", "b"], [30.0, 10.0])  # unbounded scale
    dense = _hits("dense", ["b", "a"], [0.9, 0.8])  # cosine scale
    fused = min_max_fusion([bm25, dense])
    assert {h.chunk.chunk_id: h.score for h in fused} == pytest.approx({"a": 0.5, "b": 0.5})


class LengthReranker:
    name = "rerank"

    def score(self, query, passages):
        return [float(len(p)) for p in passages]


def test_rerank_reorders_keeps_ranks_and_truncates():
    candidates = _hits("fused", ["a", "bbbb", "cc"])
    out = rerank(LengthReranker(), "q", candidates, k=2)
    assert [h.chunk.chunk_id for h in out] == ["bbbb", "cc"]
    assert out[0].ranks == {"fused": 2, "rerank": 1}
    assert rerank(LengthReranker(), "q", [], k=3) == []
