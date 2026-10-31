"""Answer generation interface and prompt construction."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from hybrid_rag.types import ScoredChunk

SYSTEM_PROMPT = """\
You answer questions about machine-learning research papers using only the numbered \
sources provided by the user.

- Base every claim on the sources. Cite the supporting source(s) inline in square brackets \
right after the claim, like [1] or [2][3].
- If the sources don't contain the answer, say so plainly instead of guessing, and mention \
what the sources do cover.
- Be concise: a direct answer first, then up to a few sentences of supporting detail.
- The sources are text extracted from PDFs and may contain extraction noise; ignore it."""

_CITATION = re.compile(r"\[(\d+)\]")


def format_context(hits: Sequence[ScoredChunk]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocks.append(f"[{i}] {c.title} (arXiv:{c.doc_id}, page {c.page})\n{c.text}")
    return "\n\n".join(blocks)


def build_user_message(question: str, hits: Sequence[ScoredChunk]) -> str:
    return f"<sources>\n{format_context(hits)}\n</sources>\n\nQuestion: {question}"


def extract_citations(text: str, n_sources: int) -> list[int]:
    """1-based source numbers cited in ``text``, in first-mention order, ignoring
    out-of-range numbers."""
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


@dataclass
class Answer:
    text: str
    model: str
    citations: list[int] = field(default_factory=list)


class Generator(Protocol):
    model: str

    def generate(self, question: str, hits: Sequence[ScoredChunk]) -> Answer: ...

    def stream(self, question: str, hits: Sequence[ScoredChunk]) -> Iterator[str]: ...
