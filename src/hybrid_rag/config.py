"""Runtime configuration, read from environment variables (prefix ``RAG_``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Data
    data_dir: Path = Path("data")
    chunks_file: Path = Path("data/processed/chunks.jsonl")
    raw_dir: Path = Path("data/raw")
    eval_file: Path = Path("data/eval/questions.jsonl")

    # Vector store: embedded on-disk Qdrant by default, a server when qdrant_url is set
    qdrant_path: Path = Path("data/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "papers"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this instruction on queries (not on documents)
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None  # None lets sentence-transformers pick cuda/mps/cpu

    # Retrieval
    candidate_k: int = 50  # depth of each first-stage retriever
    fusion: str = "rrf"  # "rrf" or "minmax" (score blending, kept for comparison)
    rrf_k: int = 60  # RRF damping constant (Cormack et al., 2009)
    rerank_depth: int = 30  # fused candidates sent to the cross-encoder
    top_k: int = 5

    # Generation (Claude). The API key is read by the SDK from ANTHROPIC_API_KEY.
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"  # grounded QA over short context doesn't need deep thinking
    llm_max_tokens: int = 4096

    # Serving
    build_index_on_startup: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
