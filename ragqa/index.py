"""Embedding + FAISS index building, saving and loading."""
import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np


@lru_cache(maxsize=2)
def get_model(name):
    from sentence_transformers import SentenceTransformer  # imported lazily (slow)

    return SentenceTransformer(name)


def embed(texts, model_name, show_progress=False):
    """Return L2-normalised float32 vectors, so inner product == cosine similarity."""
    model = get_model(model_name)
    vectors = model.encode(
        texts, batch_size=32, show_progress_bar=show_progress, normalize_embeddings=True
    )
    return np.asarray(vectors, dtype="float32")


def build_index(chunks, model_name, show_progress=True):
    vectors = embed([c["text"] for c in chunks], model_name, show_progress)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, chunks, meta, index_dir):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(index_dir):
    index_dir = Path(index_dir)
    if not (index_dir / "faiss.index").exists():
        raise FileNotFoundError(
            "No index found. Build one first: python -m ragqa.cli build --pdf data/your.pdf"
        )
    index = faiss.read_index(str(index_dir / "faiss.index"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return index, chunks, meta
