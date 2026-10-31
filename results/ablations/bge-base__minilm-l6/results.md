| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.383 | 0.617 | 0.486 | 0.37-0.60 | 0.445 | 0.584 | 129 | 139 |
| Hybrid (RRF) | 0.433 | 0.700 | 0.553 | 0.45-0.66 | 0.512 | 0.640 | 131 | 141 |
| **Hybrid + rerank** | **0.450** | **0.817** | **0.591** | 0.49-0.69 | **0.571** | **0.746** | 2380 | 2818 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.558 | 0.428 |
| Hybrid (RRF) | 0.610 | 0.506 |
| **Hybrid + rerank** | 0.669 | 0.527 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-base-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:43:24Z</sub>
