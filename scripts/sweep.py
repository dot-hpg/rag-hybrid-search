"""Sweep fusion and reranking hyper-parameters with the default models loaded once.

python scripts/sweep.py --out results/ablations/sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.eval.dataset import load_questions
from hybrid_rag.eval.runner import evaluate
from hybrid_rag.pipeline import Mode, RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/ablations/sweep.json"))
    args = parser.parse_args()

    settings = get_settings()
    questions = load_questions(settings.eval_file)
    pipeline = RAGPipeline.from_settings(settings, load_generator=False)
    rows = []

    def run(group: str, setting: str, mode: Mode) -> None:
        report = evaluate(pipeline, questions, [mode])[0]
        rows.append(
            {
                "group": group,
                "setting": setting,
                "mode": mode.value,
                "metrics": report.metrics,
                "mrr_ci95": report.mrr_ci95,
                "latency_ms": report.latency_ms,
            }
        )
        print(group, setting, mode.value, report.metrics["mrr@10"], report.latency_ms)

    for fusion, k in [("rrf", 10), ("rrf", 30), ("rrf", 60), ("rrf", 100), ("minmax", 0)]:
        pipeline.fusion, pipeline.rrf_k = fusion, k or 60
        name = f"RRF k={k}" if fusion == "rrf" else "min-max score blending"
        run("fusion", name, Mode.HYBRID)
        run("fusion", name, Mode.HYBRID_RERANK)
    pipeline.fusion, pipeline.rrf_k = settings.fusion, settings.rrf_k

    for depth in [10, 20, 30, 50]:
        pipeline.rerank_depth = depth
        run("rerank_depth", f"rerank top {depth}", Mode.HYBRID_RERANK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
