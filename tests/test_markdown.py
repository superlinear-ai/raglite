"""Test Markdown conversion."""

from pathlib import Path

from raglite._markdown import document_to_markdown


def test_pdf_with_missing_font_sizes() -> None:
    """Test conversion of a PDF with missing font sizes."""
    # Convert a PDF whose parsed font sizes are all equal to 1.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    doc = document_to_markdown(doc_path)
    # Verify that we can reconstruct the font sizes and heading levels regardless of the missing
    # font size data.
    expected_heading = """
# ON THE ELECTRODYNAMICS OF MOVING BODIES

## By A. EINSTEIN June 30, 1905

It is known that Maxwell
    """.strip()
    assert doc.startswith(expected_heading)
