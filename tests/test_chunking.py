import pytest

from hybrid_rag.ingest.chunking import (
    build_chunks,
    chunk_words,
    clean,
    is_reference_chunk,
    reference_density,
)

PROSE = (
    "Chain-of-thought prompting is an emergent ability of model scale (Wei et al., 2022b). "
    "That is, it does not positively impact performance for small models, and only yields "
    "gains when used with models of roughly 100B parameters (Brown et al., 2020). We found "
    "that smaller models produced fluent but illogical chains of thought, leading to lower "
    "performance than standard prompting on GSM8K and other arithmetic benchmarks."
)
BIBLIOGRAPHY = (
    "Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly "
    "supervised open domain question answering. In Proceedings of ACL. Mina Lee, Percy Liang, "
    "and Qian Yang. 2022. CoAuthor: Designing a human-AI collaborative writing dataset. In "
    "Proceedings of the CHI Conference on Human Factors in Computing Systems. Patrick Lewis, "
    "Ethan Perez, and Aleksandra Piktus. 2020. Retrieval-augmented generation for "
    "knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems. "
    "URL https://arxiv.org/abs/2005.11401."
)


def test_clean_rejoins_hyphenation_and_splits_ligatures():
    assert clean("efﬁcient trans-\nformer   models\n") == "efficient transformer models"


def test_chunk_words_windows_overlap_and_pages():
    words = [(f"w{i}", 1 + i // 100) for i in range(450)]
    windows = list(chunk_words(words, size=200, overlap=40, min_words=50))
    # starts at 0, 160, 320; the window at 480 would be empty
    assert [w.split()[0] for w, _ in windows] == ["w0", "w160", "w320"]
    assert [page for _, page in windows] == [1, 2, 4]
    first, second = windows[0][0].split(), windows[1][0].split()
    assert first[-40:] == second[:40]


def test_chunk_words_drops_short_tail():
    words = [(f"w{i}", 1) for i in range(200 + 30)]
    windows = list(chunk_words(words, size=200, overlap=40, min_words=80))
    assert len(windows) == 1


@pytest.mark.parametrize("overlap", [-1, 200, 250])
def test_chunk_words_rejects_bad_overlap(overlap):
    with pytest.raises(ValueError):
        list(chunk_words([("a", 1)], size=200, overlap=overlap))


def test_reference_filter_separates_bibliography_from_cited_prose():
    assert reference_density(PROSE) < 1.0
    assert not is_reference_chunk(PROSE)
    assert is_reference_chunk(BIBLIOGRAPHY)


def test_build_chunks_ids_are_stable_when_filtering():
    pages = [(1, PROSE * 6), (2, BIBLIOGRAPHY * 6)]
    kept = build_chunks("1234.5678", "Paper", pages, drop_references=True)
    everything = build_chunks("1234.5678", "Paper", pages, drop_references=False)
    assert 0 < len(kept) < len(everything)
    by_id = {c.chunk_id: c for c in everything}
    assert all(by_id[c.chunk_id] == c for c in kept)
    assert kept[0].chunk_id == "1234.5678_0" and kept[0].page == 1
