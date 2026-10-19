"""Test Markdown conversion."""

from pathlib import Path

import pytest

from raglite._markdown import document_to_markdown


def test_pdf_with_missing_font_sizes() -> None:
    """Test conversion of a PDF with missing font sizes."""
    # Convert a PDF whose parsed font sizes are all equal to 1.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    doc = document_to_markdown(doc_path)
    # Verify that we can reconstruct the font sizes and heading levels regardless of the missing
    # font size data.
    expected_heading = "# ON THE ELECTRODYNAMICS OF MOVING  BODIES \n\n## By A. EINSTEIN  June 30, 1905 \n\nIt is known that Maxwell"
    assert doc.startswith(expected_heading)


@pytest.mark.usefixtures("cp1252_default_text_encoding")
@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_text_file_is_read_as_utf8(tmp_path: Path, suffix: str) -> None:
    """Test that Markdown and plain text files are read as UTF-8 regardless of the locale."""
    expected_doc = "# Note\n\nHe said “hello” — it’s fine.\n"  # noqa: RUF001
    doc_path = tmp_path / f"note{suffix}"
    doc_path.write_bytes(expected_doc.encode("utf-8"))
    assert document_to_markdown(doc_path) == expected_doc


@pytest.mark.usefixtures("cp1252_default_text_encoding")
def test_unsupported_format_fallback_is_read_as_utf8(tmp_path: Path) -> None:
    """Test that files that pandoc cannot convert are read as UTF-8 regardless of the locale."""
    pytest.importorskip("pypandoc")
    expected_doc = "He said “hello” — it’s fine.\n"  # noqa: RUF001
    doc_path = tmp_path / "note.unsupported"
    doc_path.write_bytes(expected_doc.encode("utf-8"))
    assert document_to_markdown(doc_path) == expected_doc
