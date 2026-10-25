import json
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from src.rag.papers import PAPERS

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/chunks.jsonl")
CHUNK_WORDS = 200
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 50


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # rejoin words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def load_words(pdf_path: Path):
    """Return a list of (word, page_number) for the whole PDF."""
    reader = PdfReader(str(pdf_path))
    words = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        words.extend((w, page_no) for w in text.split())
    return words


def chunk_words(words):
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        window = words[start : start + CHUNK_WORDS]
        if len(window) < MIN_CHUNK_WORDS:
            break
        yield " ".join(w for w, _ in window), window[0][1]


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for arxiv_id, title in tqdm(PAPERS.items(), desc="Ingesting"):
            pdf = RAW_DIR / f"{arxiv_id}.pdf"
            if not pdf.exists():
                print(f"missing {pdf}, skipping")
                continue
            for n, (text, page) in enumerate(chunk_words(load_words(pdf))):
                record = {
                    "chunk_id": f"{arxiv_id}_{n}",
                    "doc_id": arxiv_id,
                    "title": title,
                    "page": page,
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    print(f"Wrote {total} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()