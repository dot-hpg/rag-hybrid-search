| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.217 | 0.450 | 0.323 | 0.23-0.42 | 0.310 | 0.453 | 28 | 58 |
| Hybrid (RRF) | 0.350 | 0.617 | 0.470 | 0.36-0.58 | 0.449 | 0.603 | 26 | 30 |
| **Hybrid + rerank** | **0.450** | **0.817** | **0.594** | 0.49-0.70 | **0.566** | **0.736** | 2504 | 2813 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.329 | 0.318 |
| Hybrid (RRF) | 0.531 | 0.419 |
| **Hybrid + rerank** | 0.667 | 0.535 |

<sub>60 labelled questions over 2025 chunks · embeddings `sentence-transformers/all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:26:52Z</sub>
