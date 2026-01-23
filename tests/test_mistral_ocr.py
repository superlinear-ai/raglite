"""Test MistralOCR integration."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from raglite import ImageType, MistralOCRConfig, RAGLiteConfig
from raglite._markdown import document_to_markdown
from raglite._mistral_ocr import (
    SUPPORTED_EXTENSIONS,
    _get_api_key,
    _process_ocr_response,
    mistral_ocr_to_markdown,
)


class TestMistralOCRConfig:
    """Test MistralOCR configuration."""

    def test_default_processor_is_none(self) -> None:
        """Default processor should be None (use default processor)."""
        config = RAGLiteConfig()
        assert config.document_processor is None

    def test_mistral_processor_config(self) -> None:
        """Can configure MistralOCR processor."""
        processor_config = MistralOCRConfig(
            api_key="test-key",
            include_image_descriptions=False,
        )
        config = RAGLiteConfig(document_processor=processor_config)
        assert config.document_processor is not None
        assert config.document_processor.api_key == "test-key"
        assert config.document_processor.include_image_descriptions is False

    def test_default_image_descriptions_enabled(self) -> None:
        """Image descriptions should be enabled by default."""
        processor_config = MistralOCRConfig()
        assert processor_config.include_image_descriptions is True

    def test_exclude_image_types_default_empty(self) -> None:
        """Exclude image types should be empty by default."""
        processor_config = MistralOCRConfig()
        assert processor_config.exclude_image_types == frozenset()

    def test_exclude_image_types_can_be_set(self) -> None:
        """Can set exclude image types."""
        processor_config = MistralOCRConfig(
            exclude_image_types=frozenset({ImageType.LOGO, ImageType.ICON})
        )
        assert ImageType.LOGO in processor_config.exclude_image_types
        assert ImageType.ICON in processor_config.exclude_image_types
        assert ImageType.PHOTO not in processor_config.exclude_image_types


class TestMistralOCRAPIKey:
    """Test MistralOCR API key handling."""

    def test_api_key_from_config(self) -> None:
        """API key can be provided via config."""
        processor_config = MistralOCRConfig(api_key="config-key")
        assert _get_api_key(processor_config) == "config-key"

    def test_api_key_from_env(self) -> None:
        """API key falls back to environment variable."""
        processor_config = MistralOCRConfig()
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "env-key"}):
            assert _get_api_key(processor_config) == "env-key"

    def test_api_key_config_takes_precedence(self) -> None:
        """Config API key takes precedence over environment variable."""
        processor_config = MistralOCRConfig(api_key="config-key")
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "env-key"}):
            assert _get_api_key(processor_config) == "config-key"

    def test_missing_api_key_raises_error(self) -> None:
        """Missing API key should raise ValueError."""
        processor_config = MistralOCRConfig()
        with patch.dict(os.environ, {}, clear=True):
            # Remove MISTRAL_API_KEY if present.
            os.environ.pop("MISTRAL_API_KEY", None)
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                _get_api_key(processor_config)


class TestMistralOCRFallback:
    """Test MistralOCR fallback behavior."""

    def test_unsupported_extension_falls_back(self) -> None:
        """Unsupported file extensions should fall back to default processor."""
        # .odt is not in SUPPORTED_EXTENSIONS.
        assert ".odt" not in SUPPORTED_EXTENSIONS

        config = RAGLiteConfig(document_processor=MistralOCRConfig(api_key="test-key"))

        with patch("raglite._markdown._default_document_to_markdown") as mock_default:
            mock_default.return_value = "fallback content"
            result = document_to_markdown(Path("test.odt"), config=config)
            mock_default.assert_called_once_with(Path("test.odt"))
            assert result == "fallback content"


class TestMistralOCRMocked:
    """Test MistralOCR with mocked API."""

    def test_process_ocr_response_basic(self) -> None:
        """Test _process_ocr_response with basic markdown."""
        # Create mock OCR response.
        mock_page = MagicMock()
        mock_page.markdown = "# Test Document\n\nThis is a test."
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        result = _process_ocr_response(mock_response, include_image_descriptions=False)

        assert "# Test Document" in result
        assert "This is a test." in result

    def test_process_ocr_response_with_annotations(self) -> None:
        """Test _process_ocr_response replaces image placeholders with annotations."""
        # Create mock image with valid JSON annotation.
        mock_image = MagicMock()
        mock_image.id = "img-0.jpeg"
        mock_image.image_annotation = (
            '{"image_type": "diagram", "description": "A flowchart showing the process"}'
        )

        mock_page = MagicMock()
        mock_page.markdown = "# Document\n\n![img-0.jpeg](img-0.jpeg)\n\nSome text."
        mock_page.images = [mock_image]

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        result = _process_ocr_response(mock_response, include_image_descriptions=True)

        # Image placeholder should be replaced with annotation.
        assert "![img-0.jpeg](img-0.jpeg)" not in result
        assert "[Image (diagram): A flowchart showing the process]" in result

    def test_process_ocr_response_with_raw_annotation_fallback(self) -> None:
        """Test _process_ocr_response falls back to raw annotation if parsing fails."""
        # Create mock image with non-JSON annotation.
        mock_image = MagicMock()
        mock_image.id = "img-0.jpeg"
        mock_image.image_annotation = "diagram: A flowchart showing the process"

        mock_page = MagicMock()
        mock_page.markdown = "# Document\n\n![img-0.jpeg](img-0.jpeg)\n\nSome text."
        mock_page.images = [mock_image]

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        result = _process_ocr_response(mock_response, include_image_descriptions=True)

        # Should fall back to raw annotation.
        assert "[Image: diagram: A flowchart showing the process]" in result

    def test_process_ocr_response_multiple_pages(self) -> None:
        """Test _process_ocr_response with multiple pages."""
        mock_page1 = MagicMock()
        mock_page1.markdown = "# Page 1"
        mock_page1.images = []

        mock_page2 = MagicMock()
        mock_page2.markdown = "# Page 2"
        mock_page2.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page1, mock_page2]

        result = _process_ocr_response(mock_response, include_image_descriptions=False)

        assert "# Page 1" in result
        assert "# Page 2" in result
        # Pages should be separated by double newlines.
        assert "\n\n" in result

    def test_process_ocr_response_exclude_image_types(self) -> None:
        """Test _process_ocr_response excludes specified image types."""
        # Create mock images with different types.
        mock_logo = MagicMock()
        mock_logo.id = "img-logo.jpeg"
        mock_logo.image_annotation = '{"image_type": "logo", "description": "Company logo"}'

        mock_diagram = MagicMock()
        mock_diagram.id = "img-diagram.jpeg"
        mock_diagram.image_annotation = (
            '{"image_type": "diagram", "description": "Architecture diagram"}'
        )

        mock_page = MagicMock()
        mock_page.markdown = (
            "# Document\n\n![](img-logo.jpeg)\n\n![](img-diagram.jpeg)\n\nSome text."
        )
        mock_page.images = [mock_logo, mock_diagram]

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        result = _process_ocr_response(
            mock_response,
            include_image_descriptions=True,
            exclude_image_types=frozenset({ImageType.LOGO}),
        )

        # Logo should be removed entirely.
        assert "logo" not in result.lower()
        assert "Company logo" not in result
        # Diagram should be included.
        assert "[Image (diagram): Architecture diagram]" in result


class TestDocumentToMarkdownDispatch:
    """Test document_to_markdown dispatcher."""

    def test_default_processor_used(self) -> None:
        """Default config should use default processor."""
        config = RAGLiteConfig()

        with patch("raglite._markdown._default_document_to_markdown") as mock_default:
            mock_default.return_value = "default content"
            # Need to create a real file for this test.
            document_to_markdown(Path("test.txt"), config=config)
            mock_default.assert_called_once_with(Path("test.txt"))

    def test_mistral_processor_used(self) -> None:
        """Mistral config should use MistralOCR processor."""
        processor_config = MistralOCRConfig(api_key="test-key")
        config = RAGLiteConfig(document_processor=processor_config)

        # Patch at the source module where mistral_ocr_to_markdown is defined.
        with patch("raglite._mistral_ocr.mistral_ocr_to_markdown") as mock_mistral:
            mock_mistral.return_value = "mistral content"
            result = document_to_markdown(Path("test.pdf"), config=config)
            mock_mistral.assert_called_once_with(
                Path("test.pdf"), processor_config=processor_config
            )
            assert result == "mistral content"


@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set")
@pytest.mark.slow
class TestMistralOCRIntegration:
    """Integration tests requiring actual API access."""

    def test_real_pdf_conversion(self) -> None:
        """Test real PDF conversion with actual API."""
        processor_config = MistralOCRConfig(include_image_descriptions=True)

        # Use the test PDF that exists in the tests directory.
        doc_path = Path(__file__).parent / "specrel.pdf"
        if not doc_path.exists():
            pytest.skip("Test PDF not found")

        result = mistral_ocr_to_markdown(doc_path, processor_config=processor_config)

        # Basic sanity checks.
        assert len(result) > 100  # noqa: PLR2004
        assert isinstance(result, str)
