"""Split pages into overlapping word windows while keeping the page number."""


def chunk_pages(pages, chunk_words=180, overlap=40):
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words")

    step = chunk_words - overlap
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            chunks.append({"id": len(chunks), "page": page["page"], "text": " ".join(window)})
            if start + chunk_words >= len(words):
                break
    return chunks
