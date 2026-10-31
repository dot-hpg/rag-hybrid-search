# Deployment

The Docker image is self-contained: it downloads both models and builds the vector index
at build time, then serves the API and web UI with uvicorn on `$PORT` (default 7860).
It runs on any container host. Memory use is ~1.2 GB; 2 vCPU is plenty for a demo.

## Hugging Face Spaces (free, recommended for a portfolio demo)

Automatic deploys from GitHub on every push to `main`:

1. Create a Space at <https://huggingface.co/new-space>: pick **Docker** as the SDK and
   **Blank** as the template. Note its id, e.g. `your-hf-username/rag-hybrid-search`.
2. Create a Hugging Face access token with **write** permission
   (<https://huggingface.co/settings/tokens>).
3. In this GitHub repo: **Settings → Secrets and variables → Actions**
   * Secrets → New repository secret: `HF_TOKEN` = the token.
   * Variables → New repository variable: `HF_SPACE` = the Space id.
4. Run **Actions → Deploy demo to Hugging Face Spaces → Run workflow** (or push to `main`).
   The first build takes ~10 minutes; the Space page shows the build log.
5. Optional: in the Space's **Settings → Variables and secrets**, add `ANTHROPIC_API_KEY`
   as a secret to switch answers from extractive to Claude.

The workflow copies the app into the Space and prepends the Space metadata
(`sdk: docker`, `app_port: 7860`) to the README. Free Spaces sleep after 48 hours without
traffic and wake on the next visit (~30 s).

## Docker, locally

```bash
docker build -t rag-hybrid-search .
docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY rag-hybrid-search
# open http://localhost:7860
```

On Windows PowerShell, `-e ANTHROPIC_API_KEY` passes the variable through if you set it
first with `$env:ANTHROPIC_API_KEY = "sk-ant-..."`.

## Docker Compose with a Qdrant server

```bash
docker compose up --build
# API + UI on http://localhost:8000, Qdrant dashboard on http://localhost:6333/dashboard
```

The API container builds the collection in Qdrant on first start and reuses it after.

## Render / Fly.io / Cloud Run

Point the service at the Dockerfile. All three set `$PORT`, which the image honours.
Set the health check path to `/health`, and give it at least 1 GB of memory.

## Configuration

All settings are environment variables with the `RAG_` prefix (see `.env.example` and
`src/hybrid_rag/config.py`). The ones you're most likely to change:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude answers; otherwise extractive |
| `RAG_QDRANT_URL` | unset | Use a Qdrant server instead of the embedded index |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any sentence-transformers cross-encoder |
| `RAG_RERANK_DEPTH` | `30` | Candidates the cross-encoder scores; latency scales linearly |
| `RAG_LLM_MODEL` | `claude-opus-5-5` | Claude model for answers |
| `RAG_LLM_EFFORT` | `low` | `low` / `medium` / `high`: answer depth vs latency |
