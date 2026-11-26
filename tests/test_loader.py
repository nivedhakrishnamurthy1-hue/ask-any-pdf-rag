from ragqa.loader import clean_text


def test_joins_hyphenated_line_breaks():
    assert clean_text("inter-\nnational rules") == "international rules"


def test_collapses_whitespace_and_soft_hyphens():
    assert clean_text("a   b\n\n c\u00add") == "a b cd"


import io

import pytest
from pypdf import PdfWriter

from ragqa.loader import load_pdf


def blank_pdf(pages):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def test_page_limit_is_enforced():
    with pytest.raises(ValueError, match="limit"):
        load_pdf(blank_pdf(3), max_pages=2)


def test_pdf_without_text_is_rejected():
    with pytest.raises(ValueError, match="No extractable text"):
        load_pdf(blank_pdf(1))
