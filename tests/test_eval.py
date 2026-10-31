import json
from collections import Counter
from pathlib import Path

import pytest

from hybrid_rag.corpus import PAPERS, load_chunks
from hybrid_rag.eval.dataset import EvalQuestion, load_questions
from hybrid_rag.eval.metrics import (
    first_relevant_rank,
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from hybrid_rag.eval.runner import evaluate, render_markdown, results_payload, update_readme
from hybrid_rag.ingest.chunking import is_reference_chunk
from hybrid_rag.pipeline import Mode

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data/processed/chunks.jsonl"
QUESTIONS_FILE = ROOT / "data/eval/questions.jsonl"


def test_metrics():
    ranked, rel = ["a", "b", "c", "d"], {"c", "z"}
    assert hit_at_k(ranked, rel, 2) == 0.0 and hit_at_k(ranked, rel, 3) == 1.0
    assert reciprocal_rank(ranked, rel, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, rel, 2) == 0.0
    assert recall_at_k(ranked, rel, 10) == 0.5
    assert ndcg_at_k(["c", "z"], rel, 10) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, rel, 10) == pytest.approx((1 / 2) / (1 + 1 / 1.584962500721156))
    assert first_relevant_rank(ranked, rel) == 3 and first_relevant_rank(ranked, {"q"}) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 95) == 0.0


def test_corpus_file_is_clean():
    chunks = load_chunks(CHUNKS_FILE)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.doc_id for c in chunks} == set(PAPERS)
    assert not any(is_reference_chunk(c.text) for c in chunks)
    assert not any("ﬁ" in c.text for c in chunks)  # ligatures normalised


def test_every_question_has_relevant_evidence_in_corpus():
    chunks = load_chunks(CHUNKS_FILE)
    questions = load_questions(QUESTIONS_FILE)
    assert len(questions) >= 50
    assert len({q.id for q in questions}) == len(questions)
    assert {q.kind for q in questions} == {"lexical", "semantic"}
    assert set(Counter(q.doc_id for q in questions)) == set(PAPERS)  # every paper covered
    for q in questions:
        for phrase in q.evidence:  # each phrase must be findable on its own
            single = EvalQuestion(q.id, q.question, q.doc_id, (phrase,), q.kind)
            assert single.relevant_chunk_ids(chunks), f"{q.id}: {phrase!r} not in {q.doc_id}"


def test_runner_end_to_end(pipeline, tmp_path):
    questions = [
        EvalQuestion(
            "t1",
            "What does LoRA inject into each layer?",
            "2106.09685",
            ("rank decomposition",),
            "lexical",
        ),
        EvalQuestion(
            "t2",
            "Where should relevant info sit in a long prompt?",
            "2307.03172",
            ("beginning or end",),
            "semantic",
        ),
    ]
    reports = evaluate(pipeline, questions, [Mode.BM25, Mode.HYBRID_RERANK], warmup=1)
    assert [r.mode for r in reports] == ["bm25", "hybrid_rerank"]
    assert all(0.0 <= v <= 1.0 for r in reports for v in r.metrics.values())
    assert set(reports[0].by_kind) == {"lexical", "semantic"}

    payload = results_payload(reports, questions, {"chunks": 6, "candidate_k": 10})
    json.dumps(payload)  # serialisable
    table = render_markdown(payload)
    assert "| BM25 |" in table and "MRR@10 by question type" in table

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    assert update_readme(readme, table)
    text = readme.read_text()
    assert "old" not in text and "| BM25 |" in text and text.endswith("outro\n")


def test_runner_rejects_unlabelled_questions(pipeline):
    bad = [EvalQuestion("x", "q", "2106.09685", ("not in any chunk",), "lexical")]
    with pytest.raises(ValueError, match="no relevant chunk"):
        evaluate(pipeline, bad, [Mode.BM25])


def test_bootstrap_ci_brackets_the_mean():
    from hybrid_rag.eval.runner import bootstrap_ci

    values = [0.0, 1.0] * 30
    lo, hi = bootstrap_ci(values)
    assert lo < 0.5 < hi and lo > 0.3 and hi < 0.7
    assert bootstrap_ci([1.0, 1.0]) == (1.0, 1.0)
