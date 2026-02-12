"""Test MistralOCR document processing."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from raglite import MistralOCRConfig
from raglite._mistral_ocr import (
    _build_image_annotation_model,
    _process_ocr_response,
    mistral_ocr_to_markdown,
)


def _mock_ocr_response(pages: list[tuple[str, list[tuple[str, str | None]]]]) -> SimpleNamespace:
    """Create a mock OCR response."""
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                markdown=markdown,
                images=[SimpleNamespace(id=img_id, image_annotation=ann) for img_id, ann in images],
            )
            for markdown, images in pages
        ]
    )


def test_process_ocr_response() -> None:
    """Test OCR response processing: annotations, filtering, multi-page join."""
    diagram_ann = '{"image_type": "diagram", "description": "A flowchart"}'
    logo_ann = '{"image_type": "logo", "description": "Company logo"}'
    response = _mock_ocr_response(
        [
            (
                "![](img-d.jpeg)\n\n![](img-l.jpeg)",
                [
                    ("img-d.jpeg", diagram_ann),
                    ("img-l.jpeg", logo_ann),
                ],
            ),
            ("![](img-r.jpeg)", [("img-r.jpeg", "raw fallback text")]),  # page 2
        ]
    )
    annotation_model = _build_image_annotation_model(frozenset({"diagram", "logo"}))
    result = _process_ocr_response(
        response,
        annotation_model=annotation_model,
        include_image_descriptions=True,
        exclude_image_types=frozenset({"logo"}),
    )
    assert "[Image (diagram): A flowchart]" in result
    assert "Company logo" not in result
    assert "[Image: raw fallback text]" in result  # fallback + page 2 joined


@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set")
@pytest.mark.slow
def test_real_pdf_conversion() -> None:
    """Test Mistral OCR on NVIDIA report with tables, charts, and financial data."""
    doc_path = Path(__file__).parent / "NVIDIA-report.pdf"
    result = mistral_ocr_to_markdown(
        doc_path,
        processor_config=MistralOCRConfig(include_image_descriptions=True),
    )
    assert len(result) > 500  # noqa: PLR2004  # substantial multi-page content
    assert "| " in result  # tables rendered as markdown
    assert "[Image (" in result  # image descriptions with type classification
    assert "$130.5 billion" in result  # financial data from table cells
