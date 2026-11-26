"""End-to-end test without downloading any model: a hashing bag-of-words embedder
replaces MiniLM, so build -> save -> load -> search -> evaluate is fully exercised."""
import re
import zlib

import numpy as np
import pytest

from ragqa import index as index_mod
from ragqa.chunking import chunk_pages
from ragqa.evaluation import evaluate, summarize
from ragqa.retriever import Retriever

DIM = 256


def fake_embed(texts, model_name, show_progress=False):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            out[row, zlib.crc32(word.encode()) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


PAGES = [
    {"page": 1, "text": "The library is open from eight in the morning until eight in the evening every weekday."},
    {"page": 2, "text": "Hostel fees must be paid before the first of July and late payment attracts a fine."},
    {"page": 3, "text": "Students need seventy five percent attendance to be eligible for the semester examination."},
]


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    monkeypatch.setattr(index_mod, "embed", fake_embed)
    chunks = chunk_pages(PAGES, chunk_words=50, overlap=10)
    index = index_mod.build_index(chunks, "fake", show_progress=False)
    meta = {"embed_model": "fake", "source": "test.pdf"}
    index_mod.save_index(index, chunks, meta, tmp_path)
    return Retriever.from_dir(tmp_path)


def test_search_returns_the_right_page(retriever):
    hits = retriever.search("what attendance is needed for the examination", k=2)
    assert hits[0]["page"] == 3
    assert hits[0]["score"] >= hits[1]["score"]


def test_evaluate_without_llm(retriever):
    items = [
        {"id": 1, "question": "when is the library open", "expected_pages": [1], "answerable": True},
        {"id": 2, "question": "hostel fee deadline", "expected_pages": [2], "answerable": True},
    ]
    rows = evaluate(items, retriever, k=2, min_score=0.1, use_llm=False)
    summary = summarize(rows)
    assert summary["retrieval_hit"] == 1.0
    assert summary["mrr"] == 1.0
