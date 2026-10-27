| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.200 | 0.550 | 0.344 | 0.25-0.44 | 0.336 | 0.488 | 31 | 113 |
| Hybrid (RRF) | **0.383** | 0.617 | 0.488 | 0.37-0.59 | 0.462 | 0.606 | 30 | 39 |
| **Hybrid + rerank** | **0.383** | **0.717** | **0.513** | 0.40-0.62 | **0.517** | **0.707** | 1914 | 2230 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.436 | 0.268 |
| Hybrid (RRF) | 0.579 | 0.414 |
| **Hybrid + rerank** | 0.545 | 0.486 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-small-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:04:50Z</sub>
