from ragqa import metrics as M

HITS = [
    {"page": 5, "text": "Students need 75% attendance."},
    {"page": 9, "text": "Fees are paid in July."},
]


def test_page_based_hit_and_rank():
    item = {"expected_pages": [9]}
    assert M.retrieval_hit(HITS, item)
    assert M.reciprocal_rank(HITS, item) == 0.5


def test_keyword_fallback_when_no_pages():
    item = {"expected_keywords": ["attendance"]}
    assert M.reciprocal_rank(HITS, item) == 1.0


def test_miss():
    item = {"expected_pages": [99]}
    assert not M.retrieval_hit(HITS, item)
    assert M.reciprocal_rank(HITS, item) == 0.0


def test_answer_correct_requires_all_keywords():
    item = {"expected_keywords": ["75", "attendance"]}
    assert M.answer_correct("You need 75% attendance.", item)
    assert not M.answer_correct("You need attendance.", item)


def test_citation_correct():
    assert M.citation_correct([5, 6], {"expected_pages": [6]})
    assert not M.citation_correct([1], {"expected_pages": [6]})
    assert not M.citation_correct([1], {})


def test_mean_of_empty_is_zero():
    assert M.mean([]) == 0.0
