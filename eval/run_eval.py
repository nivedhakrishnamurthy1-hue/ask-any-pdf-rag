"""Evaluate the full pipeline:

    python -m eval.run_eval              # retrieval + LLM answers
    python -m eval.run_eval --no-llm     # retrieval only (fast, no API calls)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from ragqa import config
from ragqa.evaluation import evaluate, summarize, to_markdown
from ragqa.retriever import Retriever

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(HERE / "questions.json"))
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    path = Path(args.questions)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m eval.make_questions`, review the "
                         "draft, and save it as eval/questions.json")
    items = json.loads(path.read_text(encoding="utf-8"))

    retriever = Retriever.from_dir()
    rows = evaluate(items, retriever, args.k, args.min_score, use_llm=not args.no_llm)
    summary = summarize(rows)

    meta = retriever.meta
    settings = (f"chunk={meta.get('chunk_words')} words / overlap={meta.get('overlap')}, "
                f"top-k={args.k}, min-score={args.min_score}, "
                f"embeddings={meta.get('embed_model', '?').split('/')[-1]}"
                + ("" if args.no_llm else f", llm={config.LLM_MODEL}"))
    table = to_markdown(summary, args.k, settings)
    print("\n" + table)

    if not args.no_llm:
        print("Wrong or missed answers (read these for your error analysis):")
        for r in rows:
            if r["answerable"] and not r.get("correct"):
                print(f"- Q{r['id']}: {r['question']}\n    got: {r['answer']}")
            if not r["answerable"] and not r.get("refused"):
                print(f"- Q{r['id']} (unanswerable) was answered: {r['answer']}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "latest.md").write_text(table, encoding="utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"run_{stamp}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nSaved eval/results/latest.md")


if __name__ == "__main__":
    main()
