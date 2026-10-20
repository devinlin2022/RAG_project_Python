"""Loader tests. Page flags are pure functions; PDF tests skip without deps."""

import pytest

PROSE = "This clause describes the approval flow for supplier pricing."


def test_scan_page_is_flagged():
    pytest.importorskip("pydantic_settings")
    from rag.ingest.loader import annotate

    pages = [{"page": 1, "text": "", "chars": 0, "density": 0.0}]
    result = annotate(pages, {})

    assert result[0]["scan_like"] is True
    assert result[0]["table_like"] is False


def test_pdfplumber_table_is_flagged():
    pytest.importorskip("pydantic_settings")
    from rag.ingest.loader import annotate

    pages = [{"page": 2, "text": "x" * 500, "chars": 500, "density": 0.5}]
    result = annotate(pages, {2: 1})

    assert result[0]["tables"] == 1
    assert result[0]["table_like"] is True
    assert result[0]["scan_like"] is False


def test_columnar_text_is_flagged_as_table():
    pytest.importorskip("pydantic_settings")
    from rag.ingest.loader import annotate

    text = "\n".join(
        f"Row{i:02d}    Direct Materials    CRMC    Checked" for i in range(10)
    )
    pages = [{"page": 3, "text": text, "chars": len(text), "density": 3.0}]
    result = annotate(pages, {})

    assert result[0]["column_ratio"] >= 0.4
    assert result[0]["table_like"] is True


def test_dense_prose_is_not_marked_as_table():
    """Regression: high text density does not imply a table page."""
    pytest.importorskip("pydantic_settings")
    from rag.ingest.loader import annotate

    text = "\n".join([PROSE] * 20)
    pages = [{"page": 4, "text": text, "chars": len(text), "density": 7.4}]
    result = annotate(pages, {})

    assert result[0]["column_ratio"] == 0.0
    assert result[0]["table_like"] is False


def test_column_ratio_ignores_short_pages():
    pytest.importorskip("pydantic_settings")
    from rag.ingest.loader import column_ratio

    assert column_ratio("a    b    c") == 0.0


def test_load_pdf_rejects_invalid_file(tmp_path):
    pytest.importorskip("pdfminer")
    pytest.importorskip("pydantic_settings")
    from rag.ingest.loader import load_pdf

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a real pdf")

    with pytest.raises(ValueError):
        load_pdf(broken)


def test_load_pdf_on_sample_when_available():
    """Copy any PDF to tests/sample.pdf to verify page numbering."""
    pytest.importorskip("pdfminer")
    pytest.importorskip("pydantic_settings")
    from pathlib import Path

    sample = Path(__file__).parent / "sample.pdf"
    if not sample.exists():
        pytest.skip("tests/sample.pdf not provided")

    from rag.ingest.loader import load_pdf

    pages = load_pdf(sample)
    assert pages
    assert [p["page"] for p in pages] == list(range(1, len(pages) + 1))
