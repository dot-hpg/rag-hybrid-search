# Evaluation

## What is measured

Retrieval quality of each pipeline configuration on a fixed set of labelled questions:

| Mode | Pipeline |
|---|---|
| `bm25` | BM25 only |
| `dense` | bge-small embeddings in Qdrant |
| `hybrid` | BM25 top-50 + dense top-50, fused with RRF |
| `dense_rerank` | dense top-50, cross-encoder reranks the top 30 |
| `hybrid_rerank` | hybrid, then cross-encoder reranks the fused top 30 (the default) |

Generation is not scored here; retrieval is the part a RAG system can't recover from. If
the right passage isn't in the top 5, no prompt fixes the answer.

## The question set

[`data/eval/questions.jsonl`](../data/eval/questions.jsonl): 60 questions, two or three per paper, covering all 24.

```json
{"id": "q39", "question": "Why doesn't LoRA slow down inference the way adapter layers do?",
 "doc_id": "2106.09685", "kind": "semantic", "evidence": ["no additional inference latency"]}
```

* **`evidence`** holds verbatim spans from the passage that answers the question. A chunk
  is relevant iff it belongs to `doc_id` and contains one of them (after whitespace,
  case and Unicode normalisation). Most questions have 1-2 relevant chunks; a few concepts
  that the paper repeats (e.g. "U-shaped") have more.
* **`kind`** splits the set so the two retrievers' strengths can be compared:
  * `lexical` (27): the question contains the rare exact term that's in the passage
    (a number, acronym or name such as "SwiGLU", "28.4 BLEU", "GPT-J").
  * `semantic` (33): the question paraphrases the passage and avoids its key terms
    ("swapped for some other arbitrary word" vs "replaced with a random token").

`tests/test_eval.py` checks that every evidence phrase still matches a chunk in its paper,
that every paper is covered, and that ids are unique.

## Metrics

All are computed on the top 10 results, with binary relevance.

| Metric | Meaning |
|---|---|
| Hit@k | Share of questions with at least one relevant chunk in the top k. Hit@5 is the one that matters for RAG, since 5 chunks go into the prompt. |
| MRR@10 | Mean of 1/rank of the first relevant chunk (0 if none in the top 10). Rewards putting the answer first. |
| nDCG@10 | Position-discounted gain over all relevant chunks, normalised by the ideal ranking. |
| Recall@10 | Share of a question's relevant chunks found in the top 10. |
| Paper Hit@5 | Whether the right *paper* appears among the top 5 distinct papers (in results.json). |
| p50 / p95 ms | Retrieval latency per query on the benchmark machine (GitHub Actions, 4 vCPU, no GPU), excluding generation. |

The **95% CI** column is a percentile bootstrap (2,000 resamples) of MRR@10 over the
questions. With 60 questions, intervals are roughly ±0.1 wide, so two modes whose
intervals overlap heavily shouldn't be called different on this set alone.

## Reproduce

```bash
rag index        # once, ~1-2 min on CPU
rag eval         # all modes; writes results/results.json and results/results.md
rag eval -m bm25 -m hybrid --update-readme
```

In GitHub, the **Benchmark** workflow does the same on a clean runner (Actions tab →
Benchmark → Run workflow, or push a commit whose message contains `[bench]`) and commits
the results back to the branch.

`results/results.json` also keeps, for each mode, the rank of the first relevant chunk for
every question (`per_question`), which is the place to start error analysis:

```python
import json

r = {m["mode"]: m["per_question"] for m in json.load(open("results/results.json"))["modes"]}
fixed = [q for q in r["dense"] if r["dense"][q] is None and r["hybrid_rerank"][q] is not None]
```

## Limitations

* One annotator wrote the questions while looking at the papers, so questions tend to be
  answerable from a single passage; real users ask vaguer, multi-hop questions.
* 60 questions is enough to see large effects, not small ones (see the CIs).
* Evidence matching can miss a chunk that answers the question in different words, which
  biases scores down slightly, equally for every mode.
* Latency comes from a shared CI runner and varies between runs by roughly ±20%.
