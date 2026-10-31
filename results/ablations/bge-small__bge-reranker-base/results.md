| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (bge-small) | 0.217 | 0.667 | 0.403 | 0.32-0.50 | 0.382 | 0.545 | 52 | 71 |
| Hybrid (RRF) | 0.417 | 0.750 | 0.562 | 0.46-0.66 | 0.515 | 0.656 | 52 | 58 |
| **Hybrid + rerank** | **0.533** | **0.850** | **0.661** | 0.56-0.76 | **0.633** | **0.782** | 14643 | 16915 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| Dense (bge-small) | 0.525 | 0.303 |
| Hybrid (RRF) | 0.673 | 0.470 |
| **Hybrid + rerank** | 0.773 | 0.569 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-small-en-v1.5` · reranker `BAAI/bge-reranker-base` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:44:17Z</sub>
