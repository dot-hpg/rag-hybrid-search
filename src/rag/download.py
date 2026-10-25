import time
from pathlib import Path

import requests

from src.rag.papers import PAPERS

RAW_DIR = Path("data/raw")
HEADERS = {"User-Agent": "rag-hybrid-search-portfolio (learning project)"}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for arxiv_id, title in PAPERS.items():
        out = RAW_DIR / f"{arxiv_id}.pdf"
        if out.exists() and out.stat().st_size > 10_000:
            print(f"skip   {arxiv_id}  {title}")
            continue
        resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=HEADERS, timeout=60)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        print(f"saved  {arxiv_id}  {title}")
        time.sleep(3)  # be polite to arXiv


if __name__ == "__main__":
    main()