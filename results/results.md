| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.350 | 0.667 | 0.496 | 0.39-0.60 | 0.501 | 0.677 | 0.1 | 0.2 |
| Dense (bge-small) | 0.200 | 0.550 | 0.344 | 0.25-0.44 | 0.336 | 0.488 | 42 | 110 |
| Hybrid (RRF) | **0.383** | 0.617 | 0.488 | 0.37-0.59 | 0.462 | 0.606 | 41 | 47 |
| Dense + rerank | **0.383** | 0.667 | 0.486 | 0.37-0.60 | 0.465 | 0.596 | 2357 | 2551 |
| **Hybrid + rerank** | **0.383** | **0.717** | **0.513** | 0.40-0.62 | **0.517** | **0.707** | 2238 | 2538 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| BM25 | 0.496 | 0.497 |
| Dense (bge-small) | 0.436 | 0.268 |
| Hybrid (RRF) | 0.579 | 0.414 |
| Dense + rerank | 0.533 | 0.447 |
| **Hybrid + rerank** | 0.545 | 0.486 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-small-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:09:08Z</sub>
