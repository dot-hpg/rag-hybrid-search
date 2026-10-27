"""FastAPI service: search, compare retrieval modes, and ask for cited answers."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from hybrid_rag import __version__
from hybrid_rag.config import get_settings
from hybrid_rag.corpus import PAPERS
from hybrid_rag.pipeline import Mode, RAGPipeline, SearchResult

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1, max_length=1000, examples=["How does LoRA cut trainable parameters?"]
    )
    mode: Mode = Mode.HYBRID_RERANK
    k: int = Field(default=5, ge=1, le=20)


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


def _result_json(result: SearchResult) -> dict:
    return {
        "query": result.query,
        "mode": result.mode.value,
        "hits": [h.to_dict() for h in result.hits],
        "timings_ms": result.timings_ms,
        "total_ms": round(result.total_ms, 2),
    }


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """Build the app. Pass a ready pipeline (tests do this) or let startup load one
    from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if pipeline is None:
            logging.basicConfig(level=logging.INFO)
            app.state.pipeline = RAGPipeline.from_settings(get_settings())
        else:
            app.state.pipeline = pipeline
        yield

    app = FastAPI(
        title="Hybrid-Search RAG",
        version=__version__,
        description="BM25 + dense retrieval, reciprocal rank fusion and cross-encoder "
        "reranking over 24 ML research papers, with cited answers.",
        lifespan=lifespan,
    )

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    def run_search(request: Request, query: str, mode: Mode, k: int) -> SearchResult:
        try:
            return get_pipeline(request).search(query, mode, k)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/info")
    def info(request: Request) -> dict:
        p = get_pipeline(request)
        settings = get_settings()
        return {
            "version": __version__,
            "papers": len({c.doc_id for c in p.chunks}),
            "chunks": len(p.chunks),
            "modes": [m.value for m in p.available_modes],
            "default_mode": Mode.HYBRID_RERANK.value
            if Mode.HYBRID_RERANK in p.available_modes
            else p.available_modes[-1].value,
            "embedding_model": settings.embedding_model if p.dense else None,
            "reranker_model": settings.reranker_model if p.reranker else None,
            "generator": p.generator.model if p.generator else None,
        }

    @app.get("/api/papers")
    def papers() -> list[dict]:
        return [
            {"doc_id": k, "title": v, "url": f"https://arxiv.org/abs/{k}"}
            for k, v in PAPERS.items()
        ]

    @app.post("/api/search")
    def search(req: SearchRequest, request: Request) -> dict:
        return _result_json(run_search(request, req.query, req.mode, req.k))

    @app.post("/api/compare")
    def compare(req: CompareRequest, request: Request) -> dict:
        """Run the same query through every available retrieval mode."""
        modes = get_pipeline(request).available_modes
        return {m.value: _result_json(run_search(request, req.query, m, req.k)) for m in modes}

    @app.post("/api/ask")
    def ask(req: SearchRequest, request: Request) -> dict:
        p = get_pipeline(request)
        if p.generator is None:
            raise HTTPException(status_code=503, detail="answer generation is not configured")
        result = run_search(request, req.query, req.mode, req.k)
        answer = p.generator.generate(result.query, result.hits)
        return {
            **_result_json(result),
            "answer": answer.text,
            "citations": answer.citations,
            "model": answer.model,
        }

    @app.post("/api/ask/stream")
    def ask_stream(req: SearchRequest, request: Request) -> StreamingResponse:
        """Server-sent events: one ``sources`` event, then ``token`` events, then ``done``."""
        p = get_pipeline(request)
        if p.generator is None:
            raise HTTPException(status_code=503, detail="answer generation is not configured")
        result = run_search(request, req.query, req.mode, req.k)
        generator = p.generator

        def events() -> Iterator[str]:
            yield f"event: sources\ndata: {json.dumps(_result_json(result))}\n\n"
            try:
                for token in generator.stream(result.query, result.hits):
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
            except Exception:
                log.exception("generation failed")
                yield f"event: error\ndata: {json.dumps('Answer generation failed.')}\n\n"
            yield f"event: done\ndata: {json.dumps({'model': generator.model})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
