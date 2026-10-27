"""Answer generators: Claude when an API key is configured, extractive otherwise."""

from __future__ import annotations

import logging
import os

from hybrid_rag.config import Settings
from hybrid_rag.generation.base import Answer, Generator
from hybrid_rag.generation.extractive import ExtractiveGenerator

log = logging.getLogger(__name__)

__all__ = ["Answer", "ExtractiveGenerator", "Generator", "make_generator"]


def make_generator(settings: Settings) -> Generator:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        from hybrid_rag.generation.claude import ClaudeGenerator

        log.info("answers: Claude (%s, effort=%s)", settings.llm_model, settings.llm_effort)
        return ClaudeGenerator(settings.llm_model, settings.llm_effort, settings.llm_max_tokens)
    log.info("answers: extractive fallback (set ANTHROPIC_API_KEY to use Claude)")
    return ExtractiveGenerator()
