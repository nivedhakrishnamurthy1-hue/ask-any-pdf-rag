"""Grounded answering: retrieve -> gate -> LLM -> validate."""
import json

from ragqa import config
from ragqa.llm import complete_json

NOT_FOUND = "I could not find this in the document."

SYSTEM = """You answer questions using ONLY the context excerpts from a document.

Rules:
- If the answer is not clearly stated in the context, set "found" to false and use the answer "I could not find this in the document."
- Never use outside knowledge, even if you know the answer.
- Be concise: 1 to 3 sentences.
- "pages" must list the page numbers (integers) of the excerpts you actually used.

Respond with a JSON object only, in exactly this shape:
{"answer": string, "found": boolean, "pages": [integer]}"""


def build_context(hits):
    return "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)


def parse_answer(raw, allowed_pages):
    """Turn the model output into a validated dict. Never trust the raw JSON."""
    try:
        data = json.loads(raw)
        answer = str(data.get("answer", "")).strip()
        found = bool(data.get("found", False))
        pages = sorted({int(p) for p in data.get("pages", []) if int(p) in allowed_pages})
    except (TypeError, ValueError, AttributeError):
        return {"answer": NOT_FOUND, "found": False, "pages": [], "error": "invalid_json"}

    if not found or not answer:
        return {"answer": NOT_FOUND, "found": False, "pages": []}
    return {"answer": answer, "found": True, "pages": pages}


def ask(question, retriever, k=None, min_score=None, model=None, api_key=None):
    k = k or config.TOP_K
    min_score = config.MIN_SCORE if min_score is None else min_score
    model = model or config.LLM_MODEL

    hits = retriever.search(question, k=k)

    # Gate: if nothing is even loosely related, skip the LLM entirely.
    if not hits or hits[0]["score"] < min_score:
        return {"answer": NOT_FOUND, "found": False, "pages": [], "sources": hits, "gated": True}

    user = f"Context:\n{build_context(hits)}\n\nQuestion: {question}\n\nReply in JSON."
    raw = complete_json(SYSTEM, user, model, api_key=api_key)
    result = parse_answer(raw, {h["page"] for h in hits})
    result["sources"] = hits
    result["gated"] = False
    return result
