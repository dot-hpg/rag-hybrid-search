# Design decisions

Why the system is built the way it is, and what each choice trades away. Every number
quoted here comes from `results/results.json`, which the benchmark workflow regenerates.

## 1. Two retrievers, because each fails differently

| Query | BM25 | Dense |
|---|---|---|
| "What BLEU score does the Transformer get on WMT 2014?" | Strong: rare exact tokens (`bleu`, `wmt`, `2014`) | Weak: numbers and acronyms embed poorly |
| "Why doesn't LoRA slow down inference like adapters?" | Weak: the passage says "no additional inference latency" | Strong: paraphrase is what embeddings are for |

The benchmark tags every question as **lexical** or **semantic** so this difference is
measured, not assumed (see the "MRR@10 by question type" table in the README).

## 2. Reciprocal Rank Fusion instead of score blending

BM25 scores are unbounded and depend on query length; cosine similarities sit in a narrow
band near 0.5-0.9. Blending raw scores needs per-query normalisation, and min-max
normalisation is fragile: one outlier stretches the range and flattens everything else.

RRF (`score = Σ 1 / (k + rank)`) uses ranks only, so it needs no calibration and has one
parameter. `k = 60` is the value from Cormack et al. (2009); it keeps a document ranked
#1 by one retriever from overwhelming a document ranked #3 by both. `min_max_fusion` is
implemented too, for comparison.

## 3. A cross-encoder reranks only the top 30

A bi-encoder embeds query and passage separately, so it can't model interactions between
their tokens. A cross-encoder reads `(query, passage)` together and is far more precise,
but it costs one transformer forward pass per pair, so it can't search 2,000 chunks per
query. The pipeline therefore does cheap recall first (50 candidates from each retriever,
fused) and expensive precision last (rerank the fused top 30). Depth 30 is where the
latency stays interactive on a laptop CPU; raise `RAG_RERANK_DEPTH` on a GPU.

`cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on web search queries, not papers, so it
is a deliberately generic choice. Swapping in `BAAI/bge-reranker-base` is one env var.

## 4. Contextual chunk headers

Each chunk is indexed as `"{paper title}. {chunk text}"`. Many chunks never name their
own subject ("we freeze the pretrained weights and..."), so without the title neither
retriever can connect them to a question that does name it. The title is added at
index time only; the stored chunk text and the citations stay verbatim.

## 5. Chunking and cleaning

* **200-word windows, 40-word overlap.** Small enough that a chunk is about one idea (good
  for precision and for citation), large enough to hold a claim with its context. The
  overlap stops answers that straddle a boundary from being split in half.
* **Unicode NFKC normalisation.** PDF text contains ligatures (`ﬁ`), which split words such
  as "efﬁcient" into two tokens for BM25. About half of the original chunks had at least one.
* **Bibliography filter.** Reference lists are ~13% of windows and match author names and
  venue boilerplate instead of content. The filter counts reference-entry markers (a year
  followed by a period, venue names, URLs) per 100 words. It deliberately ignores in-text
  citations like "(Wei et al., 2022)", so related-work prose survives. Chunk ids are
  assigned before filtering, so they're stable either way.

## 6. BM25 written from scratch

`rank-bm25` scores every document in a Python loop. The implementation here keeps an
inverted index of numpy arrays and only touches the postings of the query's terms, so a
query takes ~0.1 ms. It uses Lucene's non-negative IDF so very common terms can't
subtract score. The tokenizer folds plurals but doesn't stem: Porter stemming merges
technical terms ("generative" and "generation") that are better kept apart. A unit test
checks the scores against the textbook formula.

## 7. Qdrant: embedded by default, server when configured

Embedded on-disk mode means `pip install` plus `rag index` is the whole setup, and the
Docker image can bake the index in for instant cold starts. Setting `RAG_QDRANT_URL`
switches to a Qdrant server (see `docker-compose.yml`) with no code changes, which is
what you'd run once the corpus outgrows one machine.

## 8. Grounded generation with citations

* Sources are numbered in the prompt and the system prompt requires inline `[n]` citations
  and an explicit "the sources don't say" when the answer isn't there.
* Answers stream over server-sent events, so the UI shows sources immediately and tokens
  as they arrive.
* Claude runs at `low` effort: grounded QA over five short passages doesn't need long
  deliberation, and it keeps latency and cost down (`RAG_LLM_EFFORT` changes it).
* Server-side refusal fallback is enabled, so a safety-classifier decline is retried on a
  fallback model in the same API call instead of failing the request.
* Without an API key, an extractive generator returns the best-matching retrieved sentences
  with citations. The demo works for anyone, and it can't hallucinate.

## 9. Evaluation labels by evidence, not by chunk id

A question's relevant chunks are those in the right paper that contain one of its
evidence phrases (verbatim spans of the answer). Labels survive re-chunking, anyone can
audit one with a text search, and a unit test fails if any phrase stops matching. See
[evaluation.md](evaluation.md).

## What I'd do next

* **Query-side:** HyDE or multi-query expansion for the semantic questions that still miss.
* **Fine-tune the reranker** on synthetic (question, passage) pairs generated from the corpus.
* **Answer-level evaluation:** faithfulness and citation precision with an LLM judge, next
  to the retrieval metrics.
* **Learned sparse retrieval** (SPLADE) as a third retriever, and Qdrant's native hybrid
  query API to push fusion into the database.
