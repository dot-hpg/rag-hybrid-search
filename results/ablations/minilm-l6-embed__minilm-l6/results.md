| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.167 | 0.367 | 0.259 | 0.17-0.35 | 0.265 | 0.411 | 16 | 20 |
| Hybrid (RRF) | 0.283 | 0.517 | 0.394 | 0.29-0.50 | 0.398 | 0.566 | 14 | 20 |
| **Hybrid + rerank** | **0.383** | **0.717** | **0.515** | 0.40-0.62 | **0.518** | **0.708** | 2003 | 2298 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.242 | 0.272 |
| Hybrid (RRF) | 0.413 | 0.378 |
| **Hybrid + rerank** | 0.543 | 0.493 |

<sub>60 labelled questions over 2025 chunks · embeddings `sentence-transformers/all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:01:45Z</sub>
