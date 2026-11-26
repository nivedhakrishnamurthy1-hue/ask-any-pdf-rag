"""Compare chunk sizes using retrieval metrics only (no LLM calls, no rate limits).

    python -m eval.sweep --pdf data/your.pdf --chunk-words 100 180 300
"""
import argparse
import json
from pathlib import Path

from ragqa import config
from ragqa.chunking import chunk_pages
from ragqa.index import build_index
from ragqa.loader import load_pdf
from ragqa.metrics import mean, reciprocal_rank, retrieval_hit
from ragqa.retriever import Retriever

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--questions", default=str(HERE / "questions.json"))
    parser.add_argument("--chunk-words", type=int, nargs="+", default=[100, 180, 300])
    parser.add_argument("--overlap-ratio", type=float, default=0.22)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    items = [i for i in json.loads(Path(args.questions).read_text(encoding="utf-8"))
             if i.get("answerable", True)]
    pages = load_pdf(args.pdf)

    lines = [f"| Chunk words | Chunks | Hit@{args.k} | MRR |", "|---|---|---|---|"]
    for size in args.chunk_words:
        chunks = chunk_pages(pages, size, int(size * args.overlap_ratio))
        index = build_index(chunks, config.EMBED_MODEL, show_progress=False)
        retriever = Retriever(index, chunks, config.EMBED_MODEL)
        hits = [retriever.search(i["question"], k=args.k) for i in items]
        hit = mean(retrieval_hit(h, i) for h, i in zip(hits, items))
        mrr = mean(reciprocal_rank(h, i) for h, i in zip(hits, items))
        lines.append(f"| {size} | {len(chunks)} | {hit * 100:.0f}% | {mrr:.2f} |")

    table = "\n".join(lines) + "\n"
    print("\n" + table)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sweep.md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
