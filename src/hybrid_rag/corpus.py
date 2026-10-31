"""The paper collection and chunk-file I/O."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from hybrid_rag.types import Chunk

# arXiv id -> title. 24 papers spanning transformers, LLM scaling/alignment,
# efficient fine-tuning and retrieval, so the corpus mixes overlapping vocabulary
# (hard for keyword search) with precise named entities (hard for dense search).
PAPERS: dict[str, str] = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers",
    "1907.11692": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "1910.10683": "T5: Exploring the Limits of Transfer Learning",
    "2005.14165": "GPT-3: Language Models are Few-Shot Learners",
    "2001.08361": "Scaling Laws for Neural Language Models",
    "2203.15556": "Chinchilla: Training Compute-Optimal Large Language Models",
    "2203.02155": "InstructGPT: Training Language Models to Follow Instructions",
    "2201.11903": "Chain-of-Thought Prompting",
    "2210.03629": "ReAct: Synergizing Reasoning and Acting",
    "2302.04761": "Toolformer",
    "2302.13971": "LLaMA: Open and Efficient Foundation Language Models",
    "2307.09288": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
    "2310.06825": "Mistral 7B",
    "2106.09685": "LoRA: Low-Rank Adaptation of Large Language Models",
    "2305.14314": "QLoRA: Efficient Finetuning of Quantized LLMs",
    "2205.14135": "FlashAttention",
    "1908.10084": "Sentence-BERT",
    "2004.04906": "Dense Passage Retrieval for Open-Domain QA",
    "2004.12832": "ColBERT: Efficient and Effective Passage Search",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
    "2212.10496": "HyDE: Precise Zero-Shot Dense Retrieval",
    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts",
}


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `rag ingest` first (or `rag download`).")
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def write_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
