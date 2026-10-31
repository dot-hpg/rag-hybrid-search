"""Grounded, cited answers from Claude via the Anthropic SDK."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

import anthropic

from hybrid_rag.generation.base import (
    SYSTEM_PROMPT,
    Answer,
    build_user_message,
    extract_citations,
)
from hybrid_rag.types import ScoredChunk

log = logging.getLogger(__name__)

# Server-side refusal fallback: if the model's safety classifier declines a request,
# the API re-runs it on Anthropic's recommended fallback model inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_NO_SOURCES = "I couldn't find any relevant passages in the indexed papers for that question."
_REFUSED = "The model declined to answer this question."


class ClaudeGenerator:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        max_tokens: int = 4096,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def _request(self, question: str, hits: Sequence[ScoredChunk]) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_message(question, hits)}],
            "output_config": {"effort": self.effort},
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
        }

    def generate(self, question: str, hits: Sequence[ScoredChunk]) -> Answer:
        if not hits:
            return Answer(_NO_SOURCES, self.model)
        response = self.client.beta.messages.create(**self._request(question, hits))
        if response.stop_reason == "refusal":
            return Answer(_REFUSED, response.model)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return Answer(text, response.model, extract_citations(text, len(hits)))

    def stream(self, question: str, hits: Sequence[ScoredChunk]) -> Iterator[str]:
        if not hits:
            yield _NO_SOURCES
            return
        with self.client.beta.messages.stream(**self._request(question, hits)) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield f"\n\n{_REFUSED}"
