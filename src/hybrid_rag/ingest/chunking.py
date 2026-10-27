"""Text cleaning, sliding-window chunking, and bibliography filtering."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence

from hybrid_rag.types import Chunk

CHUNK_WORDS = 200
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 50

_HYPHEN_BREAK = re.compile(r"-\n(\w)")
_WHITESPACE = re.compile(r"\s+")

# Bibliography signals. In-text citations ("(Wei et al., 2022)", "Stiennon et al. (2020)")
# put a ")" or ";" after the year; reference entries end the year with a period
# ("... Le. 2019. Title.") and carry venue boilerplate. Counting only these keeps
# related-work prose (dense with citations) out of the filter.
_ENTRY_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\.(?!\d)")
_VENUE = re.compile(
    r"\bIn Proceedings\b|\bIn Advances\b|\bIn International\b|\bIn [A-Z]{3,}\b|\bConference\b|"
    r"\bAssociation for Computational\b|\barXiv preprint\b|\barXiv:\s?\d|\bpages \d|\bURL\b|"
    r"https?://|\bJournal\b|\bTransactions\b|\bvolume \d|\bWorkshop\b"
)


def clean(text: str) -> str:
    """Normalise PDF text: NFKC (splits ligatures like "ﬁ" that would otherwise break
    tokenisation of words such as "efﬁcient"), rejoin words hyphenated across lines,
    and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def chunk_words(
    words: Sequence[tuple[str, int]],
    size: int = CHUNK_WORDS,
    overlap: int = OVERLAP_WORDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> Iterator[tuple[str, int]]:
    """Slide a ``size``-word window with ``overlap`` words of overlap over (word, page) pairs.

    Yields ``(text, start_page)``. Stops at the first window shorter than ``min_words``:
    such a tail is mostly covered by the previous window's overlap.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if len(window) < min_words:
            break
        yield " ".join(w for w, _ in window), window[0][1]


def reference_density(text: str) -> float:
    """Bibliography-style markers (entry years, venues, URLs) per 100 words."""
    n_words = max(len(text.split()), 1)
    hits = len(_ENTRY_YEAR.findall(text)) + len(_VENUE.findall(text))
    return 100.0 * hits / n_words


def is_reference_chunk(text: str, threshold: float = 3.0) -> bool:
    """Heuristic: is this window part of a reference list?

    On this corpus the median chunk scores 0 and prose rarely passes 2, while reference
    lists score 3-15. Dropping them (~13% of windows) stops BM25 from matching author
    names and venue boilerplate instead of content.
    """
    return reference_density(text) >= threshold


def build_chunks(
    doc_id: str,
    title: str,
    pages: Sequence[tuple[int, str]],
    drop_references: bool = True,
) -> list[Chunk]:
    """Turn a document's (page_number, raw_text) pages into chunks.

    Chunk ids are ``{doc_id}_{window_index}``, assigned *before* filtering so ids stay
    stable whether or not reference filtering is enabled.
    """
    words = [(w, page_no) for page_no, text in pages for w in clean(text).split()]
    chunks = []
    for i, (text, page) in enumerate(chunk_words(words)):
        if drop_references and is_reference_chunk(text):
            continue
        chunks.append(
            Chunk(chunk_id=f"{doc_id}_{i}", doc_id=doc_id, title=title, page=page, text=text)
        )
    return chunks
