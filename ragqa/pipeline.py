"""PDF -> index. Used by the CLI (saved to disk) and by the web app (kept in memory)."""
from pathlib import Path

from ragqa import config
from ragqa.chunking import chunk_pages
from ragqa.loader import load_pdf


def _build(source, name, chunk_words, overlap, model_name, max_pages, show_progress):
    from ragqa.index import build_index  # heavy imports only when needed

    chunk_words = config.CHUNK_WORDS if chunk_words is None else chunk_words
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    model_name = model_name or config.EMBED_MODEL

    pages = load_pdf(source, max_pages=max_pages)
    chunks = chunk_pages(pages, chunk_words, overlap)
    index = build_index(chunks, model_name, show_progress=show_progress)
    meta = {
        "source": name,
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_words": chunk_words,
        "overlap": overlap,
        "embed_model": model_name,
    }
    return index, chunks, meta


def build_from_pdf(pdf, chunk_words=None, overlap=None, index_dir=None, model_name=None):
    """Build an index from a PDF path and save it to disk (CLI)."""
    from ragqa.index import save_index

    index, chunks, meta = _build(pdf, Path(pdf).name, chunk_words, overlap, model_name,
                                 None, True)
    save_index(index, chunks, meta, index_dir or config.INDEX_DIR)
    return meta


def build_retriever(source, name, chunk_words=None, overlap=None, model_name=None,
                    max_pages=None):
    """Build an in-memory Retriever from a path or file-like PDF (web app)."""
    from ragqa.retriever import Retriever

    index, chunks, meta = _build(source, name, chunk_words, overlap, model_name,
                                 max_pages, False)
    return Retriever(index, chunks, meta["embed_model"], meta)
