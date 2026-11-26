"""PDF -> list of {"page": int, "text": str}."""
import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Remove soft hyphens, join words split across lines, collapse whitespace."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(source, max_pages=None):
    """`source` is a file path or a file-like object (for example an uploaded file)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        reader = PdfReader(str(path))
    else:
        reader = PdfReader(source)

    if reader.is_encrypted:
        raise ValueError("This PDF is password protected. Remove the password and try again.")

    total = len(reader.pages)
    if max_pages and total > max_pages:
        raise ValueError(f"This PDF has {total} pages; the limit is {max_pages}.")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": number, "text": text})

    if not pages:
        raise ValueError(
            "No extractable text found. This looks like a scanned PDF; run OCR on it first."
        )
    return pages
