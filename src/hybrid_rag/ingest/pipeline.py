"""Download papers from arXiv and turn the PDFs into a chunk file."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path

import requests
from pypdf import PdfReader
from tqdm import tqdm

from hybrid_rag.corpus import PAPERS, write_chunks
from hybrid_rag.ingest.chunking import build_chunks
from hybrid_rag.types import Chunk

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "rag-hybrid-search (portfolio project; github.com/dot-hpg)"}
_MIN_PDF_BYTES = 10_000


def download_papers(
    raw_dir: Path, papers: Mapping[str, str] = PAPERS, delay_s: float = 3.0
) -> list[Path]:
    """Fetch each paper's PDF from arXiv, skipping files already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for arxiv_id, title in papers.items():
        out = raw_dir / f"{arxiv_id}.pdf"
        paths.append(out)
        if out.exists() and out.stat().st_size > _MIN_PDF_BYTES:
            log.info("skip   %s  %s", arxiv_id, title)
            continue
        resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        log.info("saved  %s  %s", arxiv_id, title)
        time.sleep(delay_s)  # arXiv asks clients to rate-limit themselves
    return paths


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def ingest(
    raw_dir: Path,
    out_file: Path,
    papers: Mapping[str, str] = PAPERS,
    drop_references: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for arxiv_id, title in tqdm(papers.items(), desc="Ingesting"):
        pdf = raw_dir / f"{arxiv_id}.pdf"
        if not pdf.exists():
            log.warning("missing %s, skipping (run `rag download`)", pdf)
            continue
        chunks.extend(build_chunks(arxiv_id, title, read_pdf_pages(pdf), drop_references))
    write_chunks(chunks, out_file)
    return chunks
