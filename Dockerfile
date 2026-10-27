# syntax=docker/dockerfile:1.7
#
# Self-contained image: models are downloaded and the vector index is built at
# build time, so a container serves its first request within seconds of starting.
# Works as-is on Hugging Face Spaces (Docker SDK), Render, Fly.io, Cloud Run, etc.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    RAG_BUILD_INDEX_ON_STARTUP=false

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ml]"

COPY data/processed/chunks.jsonl data/processed/chunks.jsonl
COPY data/eval data/eval

# Bake models + index into the image.
RUN rag warmup && rag index && chown -R app:app /app /opt/hf-cache

USER app
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "uvicorn hybrid_rag.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
