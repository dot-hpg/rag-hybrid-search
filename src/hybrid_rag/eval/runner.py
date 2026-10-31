"""Run the retrieval benchmark and render the results."""

from __future__ import annotations

import json
import platform
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from hybrid_rag.eval.dataset import EvalQuestion
from hybrid_rag.eval.metrics import (
    first_relevant_rank,
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from hybrid_rag.pipeline import Mode, RAGPipeline

EVAL_DEPTH = 10
METRICS = ("hit@1", "hit@3", "hit@5", "mrr@10", "ndcg@10", "recall@10", "paper_hit@5")


@dataclass
class ModeReport:
    mode: str
    metrics: dict[str, float]
    by_kind: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    mrr_ci95: tuple[float, float] = (0.0, 0.0)
    per_question: dict[str, int | None] = field(default_factory=dict)  # qid -> first relevant rank
    top_ids: dict[str, list[str]] = field(default_factory=dict)  # qid -> top-10 chunk ids


def _score_query(
    ranked_ids: list[str], ranked_docs: list[str], q: EvalQuestion, relevant: set[str]
) -> dict[str, float]:
    k = EVAL_DEPTH
    top_docs = list(dict.fromkeys(ranked_docs))  # unique papers, rank order
    return {
        "hit@1": hit_at_k(ranked_ids, relevant, 1),
        "hit@3": hit_at_k(ranked_ids, relevant, 3),
        "hit@5": hit_at_k(ranked_ids, relevant, 5),
        "mrr@10": reciprocal_rank(ranked_ids, relevant, k),
        "ndcg@10": ndcg_at_k(ranked_ids, relevant, k),
        "recall@10": recall_at_k(ranked_ids, relevant, k),
        "paper_hit@5": float(q.doc_id in top_docs[:5]),
    }


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    With ~60 questions, differences of a few MRR points are within noise; the interval
    makes that visible instead of over-reading the table.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    lo = means[int(n_resamples * alpha / 2)]
    hi = means[min(int(n_resamples * (1 - alpha / 2)), n_resamples - 1)]
    return (round(lo, 4), round(hi, 4))


def _mean(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    return {m: round(statistics.fmean(r[m] for r in rows), 4) for m in METRICS} if rows else {}


def evaluate(
    pipeline: RAGPipeline,
    questions: Sequence[EvalQuestion],
    modes: Sequence[Mode],
    warmup: int = 3,
) -> list[ModeReport]:
    relevant = {q.id: q.relevant_chunk_ids(pipeline.chunks) for q in questions}
    unlabeled = [qid for qid, rel in relevant.items() if not rel]
    if unlabeled:
        raise ValueError(f"questions with no relevant chunk in the corpus: {unlabeled}")

    reports = []
    for mode in modes:
        for q in questions[:warmup]:  # load model weights / JIT before timing
            pipeline.search(q.question, mode, EVAL_DEPTH)
        rows: list[tuple[EvalQuestion, dict[str, float]]] = []
        latencies, per_question, top_ids = [], {}, {}
        for q in tqdm(questions, desc=f"eval {mode.value}", leave=False):
            result = pipeline.search(q.question, mode, EVAL_DEPTH)
            ids = [h.chunk.chunk_id for h in result.hits]
            docs = [h.chunk.doc_id for h in result.hits]
            rows.append((q, _score_query(ids, docs, q, relevant[q.id])))
            latencies.append(result.total_ms)
            per_question[q.id] = first_relevant_rank(ids, relevant[q.id])
            top_ids[q.id] = ids
        kinds = sorted({q.kind for q in questions})
        reports.append(
            ModeReport(
                mode=mode.value,
                metrics=_mean([r for _, r in rows]),
                by_kind={kd: _mean([r for q, r in rows if q.kind == kd]) for kd in kinds},
                latency_ms={
                    "p50": round(percentile(latencies, 50), 1),
                    "p95": round(percentile(latencies, 95), 1),
                },
                mrr_ci95=bootstrap_ci([r["mrr@10"] for _, r in rows]),
                per_question=per_question,
                top_ids=top_ids,
            )
        )
    return reports


def results_payload(
    reports: Sequence[ModeReport], questions: Sequence[EvalQuestion], config: dict
) -> dict:
    kinds: dict[str, int] = {}
    for q in questions:
        kinds[q.kind] = kinds.get(q.kind, 0) + 1
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "config": config,
        "n_questions": len(questions),
        "question_kinds": kinds,
        "modes": [r.__dict__ for r in reports],
    }


_LABELS = {
    "bm25": "BM25",
    "dense": "Dense (bge-small)",
    "hybrid": "Hybrid (RRF)",
    "dense_rerank": "Dense + rerank",
    "hybrid_rerank": "**Hybrid + rerank**",
}


def _ms(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def render_markdown(payload: dict) -> str:
    modes = payload["modes"]
    best = {m: max(r["metrics"][m] for r in modes) for m in METRICS}

    def cell(r: dict, m: str) -> str:
        v = f"{r['metrics'][m]:.3f}"
        return f"**{v}**" if r["metrics"][m] == best[m] else v

    cols = ["Hit@1", "Hit@5", "MRR@10", "95% CI", "nDCG@10", "Recall@10", "p50 ms", "p95 ms"]
    lines = [
        "| Retrieval | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for r in modes:
        lat = r["latency_ms"]
        lo, hi = r.get("mrr_ci95", (0.0, 0.0))
        row = [
            cell(r, "hit@1"),
            cell(r, "hit@5"),
            cell(r, "mrr@10"),
            f"{lo:.2f}-{hi:.2f}",
            cell(r, "ndcg@10"),
            cell(r, "recall@10"),
            _ms(lat["p50"]),
            _ms(lat["p95"]),
        ]
        lines.append(f"| {_LABELS.get(r['mode'], r['mode'])} | " + " | ".join(row) + " |")

    kinds = sorted(payload["question_kinds"])
    lines += [
        "",
        "MRR@10 by question type:",
        "",
        "| Retrieval | "
        + " | ".join(f"{k} (n={payload['question_kinds'][k]})" for k in kinds)
        + " |",
        "|---|" + "---:|" * len(kinds),
    ]
    for r in modes:
        lines.append(
            f"| {_LABELS.get(r['mode'], r['mode'])} | "
            + " | ".join(f"{r['by_kind'][k]['mrr@10']:.3f}" for k in kinds)
            + " |"
        )
    cfg = payload["config"]
    lines += [
        "",
        f"<sub>{payload['n_questions']} labelled questions over {cfg.get('chunks', '?')} chunks · "
        f"embeddings `{cfg.get('embedding_model')}` · reranker `{cfg.get('reranker_model')}` · "
        f"candidates {cfg.get('candidate_k')}/retriever, RRF k={cfg.get('rrf_k')}, rerank depth "
        f"{cfg.get('rerank_depth')} · latency is retrieval only, CPU "
        f"({payload['environment']['machine']}) · generated {payload['generated_at']}</sub>",
    ]
    return "\n".join(lines)


START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(readme: Path, table_md: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table_md}\n{END}{tail}", encoding="utf-8")
    return True


def write_results(payload: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "results.json", out_dir / "results.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    return json_path, md_path
