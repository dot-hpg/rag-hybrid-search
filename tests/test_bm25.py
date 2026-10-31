import math

import numpy as np
import pytest

from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.tokenize import tokenize
from hybrid_rag.types import Chunk


def test_tokenize_lowercases_drops_stopwords_and_folds_plurals():
    assert tokenize("The Transformers use queries and keys!") == ["transformer", "query", "key"]
    assert tokenize("GPT-3 has 175B params") == ["gpt", "3", "175b", "param"]
    assert tokenize("loss class") == ["loss", "class"]  # no over-stripping


def test_exact_term_match_ranks_first(chunks):
    hits = BM25Retriever(chunks).search("What is SRAM tiling?", k=3)
    assert hits[0].chunk.doc_id == "2205.14135"
    assert hits[0].ranks == {"bm25": 1}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_scores_match_reference_formula(chunks):
    bm25 = BM25Retriever(chunks, k1=1.2, b=0.75)
    query = "trainable parameters tokens"
    docs = [tokenize(c.indexed_text) for c in chunks]
    avgdl = sum(map(len, docs)) / len(docs)
    expected = []
    for d in docs:
        s = 0.0
        for term in set(tokenize(query)):
            df = sum(term in other for other in docs)
            if not df:
                continue
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = d.count(term)
            s += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * len(d) / avgdl))
        expected.append(s)
    np.testing.assert_allclose(bm25.scores(query), expected, rtol=1e-5)


def test_no_match_and_large_k(chunks):
    bm25 = BM25Retriever(chunks)
    assert bm25.search("zzzz qqqq", k=5) == []
    assert len(bm25.search("memory", k=100)) <= len(chunks)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_idf_never_negative():
    common = [Chunk(f"d_{i}", "d", "T", 1, "attention attention") for i in range(5)]
    scores = BM25Retriever(common).scores("attention")
    assert (scores >= 0).all()
