import pytest

from ragqa.chunking import chunk_pages


def pages_of(n_words, page=1):
    return [{"page": page, "text": " ".join(f"w{i}" for i in range(n_words))}]


def test_short_page_gives_one_chunk():
    chunks = chunk_pages(pages_of(100), chunk_words=180, overlap=40)
    assert len(chunks) == 1
    assert len(chunks[0]["text"].split()) == 100


def test_overlap_and_full_coverage():
    chunks = chunk_pages(pages_of(400), chunk_words=180, overlap=40)
    assert len(chunks) == 3
    assert chunks[1]["text"].split()[0] == "w140"   # starts 140 words in (180 - 40)
    assert chunks[0]["text"].split()[-40:] == chunks[1]["text"].split()[:40]  # 40 shared words
    covered = {w for c in chunks for w in c["text"].split()}
    assert covered == {f"w{i}" for i in range(400)}


def test_page_numbers_and_ids():
    pages = pages_of(250, page=3) + pages_of(50, page=4)
    chunks = chunk_pages(pages, chunk_words=100, overlap=20)
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    assert {c["page"] for c in chunks} == {3, 4}
    assert all(c["page"] == 4 for c in chunks[-1:])


def test_empty_page_is_skipped():
    assert chunk_pages([{"page": 1, "text": ""}], 100, 10) == []


def test_invalid_settings():
    with pytest.raises(ValueError):
        chunk_pages(pages_of(10), chunk_words=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_pages(pages_of(10), chunk_words=0, overlap=0)
