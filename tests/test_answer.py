import json

import pytest

from ragqa import answer as ans


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=4):
        return self.hits


HITS = [
    {"chunk_id": 0, "page": 3, "text": "The library is open from 8 AM to 8 PM.", "score": 0.61},
    {"chunk_id": 1, "page": 4, "text": "Fines are charged per day.", "score": 0.40},
]


def test_parse_valid_and_filters_unknown_pages():
    raw = json.dumps({"answer": "8 AM to 8 PM", "found": True, "pages": [3, 99]})
    out = ans.parse_answer(raw, {3, 4})
    assert out == {"answer": "8 AM to 8 PM", "found": True, "pages": [3]}


def test_parse_invalid_json_is_safe():
    out = ans.parse_answer("not json", {3})
    assert out["found"] is False and out["error"] == "invalid_json"


def test_not_found_clears_pages():
    raw = json.dumps({"answer": "whatever", "found": False, "pages": [3]})
    out = ans.parse_answer(raw, {3})
    assert out["found"] is False and out["pages"] == [] and out["answer"] == ans.NOT_FOUND


def test_gate_skips_the_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when the best score is below the gate")

    monkeypatch.setattr(ans, "complete_json", boom)
    low = [{"chunk_id": 0, "page": 1, "text": "x", "score": 0.05}]
    out = ans.ask("capital of Australia?", FakeRetriever(low), min_score=0.2)
    assert out["gated"] is True and out["found"] is False


def test_normal_flow(monkeypatch):
    seen = {}

    def fake_llm(system, user, model, **kwargs):
        seen["user"] = user
        return json.dumps({"answer": "8 AM to 8 PM", "found": True, "pages": [3]})

    monkeypatch.setattr(ans, "complete_json", fake_llm)
    out = ans.ask("When is the library open?", FakeRetriever(HITS), min_score=0.2)
    assert out["found"] and out["pages"] == [3] and not out["gated"]
    assert "[Page 3]" in seen["user"] and "When is the library open?" in seen["user"]
