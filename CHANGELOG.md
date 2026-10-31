# Changelog

## 1.0.0 (2026-10-27)

First portfolio release: a rebuild of the original single-script prototype.

### Added
- Installable `hybrid_rag` package with a `rag` CLI (`download`, `ingest`, `index`, `warmup`, `search`, `ask`, `eval`, `serve`).
- Hand-written vectorised BM25, `bge-small-en-v1.5` dense retrieval in Qdrant (embedded, in-memory or server), reciprocal rank fusion and min-max fusion.
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) of the fused top 30.
- Cited answers from Claude, streamed over server-sent events, with an extractive fallback when no API key is set.
- FastAPI service and a web UI that streams answers and compares retrieval modes side by side.
- 60-question evidence-labelled benchmark with Hit@k, MRR, nDCG and recall, bootstrap confidence intervals and per-stage latency.
- Ablations over embedders, rerankers, fusion settings and rerank depth, run in GitHub Actions.
- Ingestion improvements: Unicode NFKC normalisation, bibliography filtering and contextual chunk headers.
- 55 tests (91% coverage), CI on Python 3.11 to 3.13, a Docker image with models and index baked in, and a Hugging Face Spaces deploy workflow.
