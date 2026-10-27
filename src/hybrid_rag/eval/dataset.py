"""Evaluation questions and their relevance labels.

Each question names the paper that answers it and one or more *evidence phrases*:
short verbatim spans from the passage that contains the answer. A chunk is relevant
iff it belongs to that paper and contains an evidence phrase. Labelling by evidence
instead of by chunk id keeps the labels valid when chunking parameters change, and
makes every label auditable with a text search.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hybrid_rag.types import Chunk

_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    doc_id: str
    evidence: tuple[str, ...]
    # "lexical": answerable by matching rare exact terms (names, numbers, acronyms).
    # "semantic": paraphrased, shares little vocabulary with the source passage.
    kind: str

    def relevant_chunk_ids(self, chunks: Sequence[Chunk]) -> set[str]:
        phrases = [normalize(e) for e in self.evidence]
        return {
            c.chunk_id
            for c in chunks
            if c.doc_id == self.doc_id and any(p in normalize(c.text) for p in phrases)
        }


def load_questions(path: Path) -> list[EvalQuestion]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            questions.append(
                EvalQuestion(
                    id=d["id"],
                    question=d["question"],
                    doc_id=d["doc_id"],
                    evidence=tuple(d["evidence"]),
                    kind=d["kind"],
                )
            )
    return questions
