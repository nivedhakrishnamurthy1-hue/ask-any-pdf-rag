"""Pure evaluation metrics (no models, no network) so they are easy to unit-test."""


def hit_matches(hit, item):
    """A retrieved chunk is relevant if it is on an expected page, or (when no pages
    are given) contains an expected keyword."""
    pages = item.get("expected_pages") or []
    if pages:
        return hit["page"] in pages
    keywords = [k.lower() for k in item.get("expected_keywords", [])]
    return bool(keywords) and any(k in hit["text"].lower() for k in keywords)


def retrieval_hit(hits, item):
    return any(hit_matches(h, item) for h in hits)


def reciprocal_rank(hits, item):
    for rank, hit in enumerate(hits, start=1):
        if hit_matches(hit, item):
            return 1.0 / rank
    return 0.0


def answer_correct(answer, item):
    keywords = [k.lower() for k in item.get("expected_keywords", [])]
    return bool(keywords) and all(k in answer.lower() for k in keywords)


def citation_correct(pages, item):
    expected = set(item.get("expected_pages") or [])
    return bool(expected) and bool(expected & set(pages))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
