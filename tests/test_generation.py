from types import SimpleNamespace

from hybrid_rag.generation.base import build_user_message, extract_citations
from hybrid_rag.generation.claude import ClaudeGenerator
from hybrid_rag.generation.extractive import ExtractiveGenerator
from hybrid_rag.types import ScoredChunk


def test_extract_citations_order_dedup_and_range():
    assert extract_citations("A [2]. B [1][2]. C [9] [0].", n_sources=3) == [2, 1]


def test_user_message_numbers_sources(chunks):
    hits = [ScoredChunk(c, 1.0) for c in chunks[:2]]
    msg = build_user_message("Q?", hits)
    assert "[1] LoRA (arXiv:2106.09685, page 1)" in msg
    assert "[2] FlashAttention" in msg
    assert msg.rstrip().endswith("Question: Q?")


def test_extractive_answer_cites_matching_source(chunks):
    hits = [ScoredChunk(c, 1.0) for c in chunks]
    answer = ExtractiveGenerator().generate("How many tokens was Chinchilla trained on?", hits)
    assert "1.4 trillion tokens" in answer.text
    assert 5 in answer.citations  # chinchilla is the 5th source
    assert ExtractiveGenerator().generate("q", []).citations == []


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _client(text: str, stop_reason: str = "end_turn"):
    response = SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_claude_generator_request_and_parsing(chunks):
    client, messages = _client("LoRA injects low-rank matrices [1].")
    gen = ClaudeGenerator(model="claude-opus-5-5", effort="low", client=client)
    answer = gen.generate("What does LoRA inject?", [ScoredChunk(chunks[0], 1.0)])
    assert answer.text == "LoRA injects low-rank matrices [1]."
    assert answer.citations == [1]
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "<sources>" in call["messages"][0]["content"]


def test_claude_generator_handles_refusal_and_no_sources(chunks):
    client, messages = _client("", stop_reason="refusal")
    gen = ClaudeGenerator(client=client)
    assert "declined" in gen.generate("q", [ScoredChunk(chunks[0], 1.0)]).text
    assert "couldn't find" in gen.generate("q", []).text
    assert len(messages.calls) == 1  # no API call without sources


class FakeStream:
    def __init__(self, parts, stop_reason):
        self.text_stream = iter(parts)
        self._final = SimpleNamespace(stop_reason=stop_reason)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def test_claude_generator_streams_text(chunks):
    calls = []

    def stream(**kwargs):
        calls.append(kwargs)
        return FakeStream(["LoRA ", "freezes weights [1]."], "end_turn")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    out = "".join(ClaudeGenerator(client=client).stream("q", [ScoredChunk(chunks[0], 1.0)]))
    assert out == "LoRA freezes weights [1]."
    assert calls[0]["fallbacks"] == "default"


def test_extractive_strips_the_papers_own_citations(chunks):
    from dataclasses import replace

    noisy = replace(
        chunks[0], text="LoRA builds on low-rank ideas [15] and adapters [3, 7] to cut parameters."
    )
    answer = ExtractiveGenerator().generate("low-rank parameters", [ScoredChunk(noisy, 1.0)])
    assert "[15]" not in answer.text and "[3, 7]" not in answer.text
    assert answer.citations == [1]
