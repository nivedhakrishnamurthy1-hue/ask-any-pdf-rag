"""Run an evaluation set against the retriever (and optionally the LLM)."""
import time

from ragqa import config
from ragqa import metrics as M
from ragqa.answer import ask


def evaluate(items, retriever, k, min_score, use_llm=True, delay=1.5, model=None):
    rows = []
    for item in items:
        answerable = item.get("answerable", True)
        row = {"id": item.get("id"), "question": item["question"], "answerable": answerable}

        if use_llm:
            result = ask(item["question"], retriever, k=k, min_score=min_score,
                         model=model or config.LLM_MODEL)
            hits = result["sources"]
            row.update(answer=result["answer"], found=result["found"],
                       cited_pages=result["pages"], gated=result.get("gated", False))
            time.sleep(delay)  # stay under the free-tier rate limit
        else:
            hits = retriever.search(item["question"], k=k)

        if answerable:
            row["hit"] = M.retrieval_hit(hits, item)
            row["rr"] = M.reciprocal_rank(hits, item)
            if use_llm:
                row["correct"] = row["found"] and M.answer_correct(row["answer"], item)
                row["cited"] = M.citation_correct(row["cited_pages"], item)
        elif use_llm:
            row["refused"] = not row["found"]
        rows.append(row)
    return rows


def summarize(rows):
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval_hit": M.mean(r["hit"] for r in answerable),
        "mrr": M.mean(r["rr"] for r in answerable),
    }
    if answerable and "correct" in answerable[0]:
        summary["answer_accuracy"] = M.mean(r["correct"] for r in answerable)
        summary["citation_accuracy"] = M.mean(r["cited"] for r in answerable)
    if unanswerable and "refused" in unanswerable[0]:
        summary["correct_refusals"] = M.mean(r["refused"] for r in unanswerable)
    return summary


def to_markdown(summary, k, settings):
    pct = lambda x: f"{x * 100:.0f}%"
    lines = [
        f"Settings: {settings}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit@{k} (n={summary['n_answerable']}) | {pct(summary['retrieval_hit'])} |",
        f"| MRR | {summary['mrr']:.2f} |",
    ]
    if "answer_accuracy" in summary:
        lines.append(f"| Answer accuracy | {pct(summary['answer_accuracy'])} |")
        lines.append(f"| Citation accuracy | {pct(summary['citation_accuracy'])} |")
    if "correct_refusals" in summary:
        lines.append(
            f"| Correct refusals (n={summary['n_unanswerable']}) | {pct(summary['correct_refusals'])} |"
        )
    return "\n".join(lines) + "\n"
