"""Command-line interface: ``rag download | ingest | index | search | ask | eval | serve``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from hybrid_rag.config import get_settings
from hybrid_rag.pipeline import Mode, RAGPipeline

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Model hubs are chatty at INFO; keep our own logs readable.
    for noisy in ("httpx", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def download() -> None:
    """Download the 24 paper PDFs from arXiv into data/raw/."""
    from hybrid_rag.ingest.pipeline import download_papers

    paths = download_papers(get_settings().raw_dir)
    typer.echo(f"{len(paths)} PDFs in {get_settings().raw_dir}")


@app.command()
def ingest(
    keep_references: bool = typer.Option(False, help="Keep bibliography chunks."),
) -> None:
    """Extract, clean and chunk the PDFs into data/processed/chunks.jsonl."""
    from hybrid_rag.ingest.pipeline import ingest as run_ingest

    s = get_settings()
    chunks = run_ingest(s.raw_dir, s.chunks_file, drop_references=not keep_references)
    typer.echo(f"Wrote {len(chunks)} chunks to {s.chunks_file}")


@app.command()
def index() -> None:
    """Embed every chunk and (re)build the Qdrant collection."""
    from hybrid_rag.corpus import load_chunks
    from hybrid_rag.retrieval.dense import (
        DenseRetriever,
        SentenceTransformerEmbedder,
        make_qdrant_client,
    )

    s = get_settings()
    chunks = load_chunks(s.chunks_file)
    embedder = SentenceTransformerEmbedder(s.embedding_model, s.query_instruction, s.device)
    client = make_qdrant_client(s.qdrant_url, str(s.qdrant_path), s.qdrant_api_key)
    count = DenseRetriever(client, s.collection, embedder).build_index(chunks)
    client.close()
    typer.echo(f"Indexed {count} chunks into collection '{s.collection}'")


@app.command()
def warmup() -> None:
    """Download the embedding and reranking models into the local cache."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    s = get_settings()
    SentenceTransformer(s.embedding_model)
    CrossEncoder(s.reranker_model)
    typer.echo("Models cached.")


def _pipeline(mode: Mode, generator: bool = False) -> RAGPipeline:
    return RAGPipeline.from_settings(
        get_settings(),
        load_dense=mode.needs_dense,
        load_reranker=mode.needs_reranker,
        load_generator=generator,
    )


@app.command()
def search(
    query: str,
    mode: Mode = typer.Option(Mode.HYBRID_RERANK, "--mode", "-m"),
    k: int = typer.Option(5, "--k", "-k"),
) -> None:
    """Retrieve the top-k chunks for QUERY."""
    result = _pipeline(mode).search(query, mode, k)
    for i, hit in enumerate(result.hits, start=1):
        ranks = " ".join(f"{s}#{r}" for s, r in hit.ranks.items())
        typer.secho(
            f"[{i}] {hit.chunk.title} (p.{hit.chunk.page})  score={hit.score:.3f}  {ranks}",
            bold=True,
        )
        typer.echo(f"    {hit.chunk.text[:220]}...\n")
    typer.echo(f"timings (ms): {result.timings_ms}")


@app.command()
def ask(
    question: str,
    mode: Mode = typer.Option(Mode.HYBRID_RERANK, "--mode", "-m"),
    k: int = typer.Option(5, "--k", "-k"),
) -> None:
    """Answer QUESTION with citations to the retrieved chunks."""
    result, answer = _pipeline(mode, generator=True).ask(question, mode, k)
    typer.echo(answer.text + "\n")
    for i, hit in enumerate(result.hits, start=1):
        marker = "*" if i in answer.citations else " "
        typer.echo(
            f"{marker}[{i}] {hit.chunk.title}, p.{hit.chunk.page} (arXiv:{hit.chunk.doc_id})"
        )
    typer.echo(f"\nmodel: {answer.model}  timings (ms): {result.timings_ms}")


@app.command(name="eval")
def evaluate(
    modes: list[Mode] = typer.Option(list(Mode), "--mode", "-m", help="Repeat to pick modes."),
    out_dir: Path = typer.Option(Path("results"), help="Where results.json/.md are written."),
    update_readme: bool = typer.Option(False, help="Replace the results table in README.md."),
) -> None:
    """Benchmark retrieval modes on the labelled question set."""
    from hybrid_rag.eval.dataset import load_questions
    from hybrid_rag.eval.runner import evaluate as run_eval
    from hybrid_rag.eval.runner import render_markdown, results_payload, write_results
    from hybrid_rag.eval.runner import update_readme as patch_readme

    s = get_settings()
    questions = load_questions(s.eval_file)
    pipeline = RAGPipeline.from_settings(
        s,
        load_dense=any(m.needs_dense for m in modes),
        load_reranker=any(m.needs_reranker for m in modes),
        load_generator=False,
    )
    reports = run_eval(pipeline, questions, modes)
    config = {
        "chunks": len(pipeline.chunks),
        "embedding_model": s.embedding_model,
        "reranker_model": s.reranker_model,
        "candidate_k": s.candidate_k,
        "rrf_k": s.rrf_k,
        "rerank_depth": s.rerank_depth,
    }
    payload = results_payload(reports, questions, config)
    json_path, _ = write_results(payload, out_dir)
    table = render_markdown(payload)
    typer.echo(table)
    typer.echo(f"\nSaved {json_path}")
    if update_readme and patch_readme(Path("README.md"), table):
        typer.echo("README.md results table updated")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API and web UI."""
    import uvicorn

    uvicorn.run("hybrid_rag.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
