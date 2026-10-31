from types import SimpleNamespace

from hybrid_rag.config import Settings
from hybrid_rag.corpus import load_chunks
from hybrid_rag.ingest import pipeline as ingest_mod
from hybrid_rag.pipeline import Mode, RAGPipeline

PAGE = " ".join(f"token{i}" for i in range(260))


def test_ingest_writes_chunks_and_skips_missing(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "chunks.jsonl"
    raw.mkdir()
    (raw / "0001.pdf").write_bytes(b"%PDF stub")
    monkeypatch.setattr(ingest_mod, "read_pdf_pages", lambda _: [(1, PAGE), (2, PAGE)])
    chunks = ingest_mod.ingest(raw, out, papers={"0001": "Stub Paper", "0002": "Missing"})
    assert chunks and {c.doc_id for c in chunks} == {"0001"}
    assert load_chunks(out) == chunks


def test_download_skips_existing_and_fetches_new(tmp_path, monkeypatch):
    (tmp_path / "have.pdf").write_bytes(b"x" * 20_000)
    fetched = []

    def fake_get(url, headers, timeout):
        fetched.append(url)
        return SimpleNamespace(content=b"%PDF new", raise_for_status=lambda: None)

    monkeypatch.setattr(ingest_mod.requests, "get", fake_get)
    paths = ingest_mod.download_papers(tmp_path, {"have": "A", "new": "B"}, delay_s=0)
    assert fetched == ["https://arxiv.org/pdf/new"]
    assert [p.name for p in paths] == ["have.pdf", "new.pdf"]
    assert (tmp_path / "new.pdf").read_bytes() == b"%PDF new"


def test_pipeline_from_settings_bm25_only(tmp_path, chunks):
    from hybrid_rag.corpus import write_chunks

    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    p = RAGPipeline.from_settings(
        Settings(chunks_file=path), load_dense=False, load_reranker=False, load_generator=True
    )
    assert p.available_modes == [Mode.BM25]
    assert p.generator is not None
    assert p.search("tiling SRAM", Mode.BM25, 1).hits[0].chunk.doc_id == "2205.14135"
