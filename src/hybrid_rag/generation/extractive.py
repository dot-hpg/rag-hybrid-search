"""LLM-free fallback: answer with the retrieved sentences that best match the question.

Lets the whole system (API, UI, demo deployment) run with no API key. It never
paraphrases, so it can't hallucinate, but it also can't synthesise across sources.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from hybrid_rag.generation.base import Answer, extract_citations
from hybrid_rag.retrieval.tokenize import tokenize
from hybrid_rag.types import ScoredChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
# The papers' own numeric citations ("[15]", "[3, 7]") would be confused with ours.
_PAPER_CITATION = re.compile(r"\s*\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")


class ExtractiveGenerator:
    model = "extractive"

    def __init__(self, max_sentences: int = 3, min_words: int = 6) -> None:
        self.max_sentences = max_sentences
        self.min_words = min_words

    def generate(self, question: str, hits: Sequence[ScoredChunk]) -> Answer:
        if not hits:
            return Answer("No relevant passages found.", self.model)
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, int, str]] = []
        for src, hit in enumerate(hits, start=1):
            text = _PAPER_CITATION.sub("", hit.chunk.text)
            for pos, sentence in enumerate(_SENTENCE_SPLIT.split(text)):
                if len(sentence.split()) < self.min_words:
                    continue
                terms = set(tokenize(sentence))
                overlap = len(q_terms & terms) / (len(q_terms) or 1)
                # Prefer higher-ranked sources when overlap ties.
                candidates.append((overlap - 0.01 * src, src, pos, sentence.strip()))
        best = sorted(candidates, reverse=True)[: self.max_sentences]
        if not best:
            top = hits[0].chunk.text
            return Answer(f"{top[:400]}... [1]", self.model, [1])
        # Present picked sentences in source order so the answer reads naturally.
        best.sort(key=lambda c: (c[1], c[2]))
        text = " ".join(f"{s} [{src}]" for _, src, _, s in best)
        return Answer(text, self.model, extract_citations(text, len(hits)))

    def stream(self, question: str, hits: Sequence[ScoredChunk]) -> Iterator[str]:
        yield self.generate(question, hits).text
