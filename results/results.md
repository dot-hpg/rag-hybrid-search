| Retrieval | Hit@1 | Hit@5 | MRR@10 | 95% CI | nDCG@10 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.400 | 0.750 | 0.553 | 0.45-0.66 | 0.530 | 0.695 | 0.1 | 0.1 |
| Dense (bge-small) | 0.217 | 0.667 | 0.403 | 0.32-0.50 | 0.382 | 0.545 | 30 | 109 |
| Hybrid (RRF) | 0.417 | 0.750 | 0.562 | 0.46-0.66 | 0.515 | 0.656 | 36 | 58 |
| Dense + rerank | **0.467** | 0.750 | 0.574 | 0.47-0.69 | 0.524 | 0.653 | 2074 | 2277 |
| **Hybrid + rerank** | **0.467** | **0.817** | **0.600** | 0.50-0.71 | **0.568** | **0.735** | 1968 | 2318 |

MRR@10 by question type:

| Retrieval | lexical (n=27) | semantic (n=33) |
|---|---:|---:|
| BM25 | 0.570 | 0.538 |
| Dense (bge-small) | 0.525 | 0.303 |
| Hybrid (RRF) | 0.673 | 0.470 |
| Dense + rerank | 0.651 | 0.510 |
| **Hybrid + rerank** | 0.669 | 0.543 |

<sub>60 labelled questions over 2025 chunks · embeddings `BAAI/bge-small-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · candidates 50/retriever, RRF k=60, rerank depth 30 · latency is retrieval only, CPU (x86_64) · generated 2026-10-27T04:31:35Z</sub>
