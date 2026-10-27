| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.317 | 0.567 | 0.422 | 0.31-0.53 | 0.412 | 0.581 | 114 | 121 |
| Hybrid (RRF) | **0.383** | 0.600 | 0.492 | 0.38-0.61 | 0.473 | 0.611 | 114 | 124 |
| **Hybrid + rerank** | **0.383** | **0.733** | **0.516** | 0.41-0.62 | **0.527** | **0.725** | 2200 | 2606 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.455 | 0.396 |
| Hybrid (RRF) | 0.528 | 0.463 |
| **Hybrid + rerank** | 0.555 | 0.485 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-base-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:17:56Z</sub>
