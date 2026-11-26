"""Draft an evaluation set from random chunks of your document.

    python -m eval.make_questions -n 12

The output (eval/questions.draft.json) is only a DRAFT. Read every question, fix wrong
keywords, delete bad ones, add some of your own, then save it as eval/questions.json.
"""
import argparse
import json
import random
from pathlib import Path

from ragqa import config
from ragqa.llm import complete_json

SYSTEM = """You write evaluation questions for a document question-answering system.
Given one passage, write ONE specific factual question that can be answered ONLY from that passage.
Do not mention "the passage" or "the text" in the question.
Also give 1 or 2 short, distinctive keywords (numbers, names or key terms, copied exactly as they
appear in the passage) that a correct answer must contain.
Respond as JSON: {"question": string, "expected_keywords": [string]}"""

OFF_TOPIC = [
    "What is the capital city of Australia?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the boiling point of water at sea level in Celsius?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=12, help="number of answerable questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks_path = config.INDEX_DIR / "chunks.json"
    if not chunks_path.exists():
        raise SystemExit("Build the index first: python -m ragqa.cli build --pdf data/your.pdf")
    chunks = [c for c in json.loads(chunks_path.read_text(encoding="utf-8"))
              if len(c["text"].split()) >= 60]

    random.seed(args.seed)
    sample = random.sample(chunks, min(args.n, len(chunks)))

    items = []
    for chunk in sample:
        raw = complete_json(SYSTEM, f"Passage:\n{chunk['text']}\n\nReply in JSON.", config.LLM_MODEL)
        try:
            data = json.loads(raw)
            items.append({
                "id": len(items) + 1,
                "question": data["question"].strip(),
                "expected_keywords": [str(k) for k in data["expected_keywords"]],
                "expected_pages": [chunk["page"]],
                "answerable": True,
            })
        except (ValueError, KeyError, TypeError):
            continue

    for question in OFF_TOPIC:
        items.append({"id": len(items) + 1, "question": question, "answerable": False})

    out = Path(__file__).parent / "questions.draft.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} draft questions to {out}")
    print("Now REVIEW them by hand and save the cleaned file as eval/questions.json")


if __name__ == "__main__":
    main()
