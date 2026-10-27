| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.200 | 0.550 | 0.344 | 0.25-0.44 | 0.336 | 0.488 | 50 | 55 |
| Hybrid (RRF) | 0.383 | 0.617 | 0.488 | 0.37-0.59 | 0.462 | 0.606 | 50 | 60 |
| **Hybrid + rerank** | **0.500** | **0.783** | **0.620** | 0.51-0.72 | **0.605** | **0.756** | 14594 | 16874 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.436 | 0.268 |
| Hybrid (RRF) | 0.579 | 0.414 |
| **Hybrid + rerank** | 0.704 | 0.552 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-small-en-v1.5` · reranker `BAAI/bge-reranker-base` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:20:03Z</sub>
