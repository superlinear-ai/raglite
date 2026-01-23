"""MistralOCR document processor for RAGLite."""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path  # noqa: TC003
from typing import Any

from pydantic import BaseModel, Field

from raglite._config import ImageType, MistralOCRConfig  # noqa: TC001

logger = logging.getLogger(__name__)

# File extensions supported by MistralOCR.
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".avif", ".docx", ".pptx", ".webp"}


class MistralOCRError(Exception):
    """Error during MistralOCR processing."""


class ImageAnnotation(BaseModel):
    """Schema for vision-based image annotation."""

    image_type: ImageType = Field(
        ...,
        description=(
            "The type of the image. Must be one of: graph, chart, diagram, table, "
            "photo, screenshot, logo, icon, or other."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "A concise description of the image content. For diagrams and charts, "
            "describe what is being illustrated. For tables, summarize the data. "
            "For photos, describe the subject matter."
        ),
    )


def _get_api_key(processor_config: MistralOCRConfig) -> str:
    """Get the Mistral API key from config or environment variable."""
    api_key = processor_config.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        error_msg = (
            "MISTRAL_API_KEY environment variable is not set and MistralOCRConfig.api_key is None."
        )
        raise ValueError(error_msg)
    return api_key


def _get_mistral_client(processor_config: MistralOCRConfig) -> Any:
    """Get a Mistral client instance."""
    try:
        from mistralai import Mistral
    except ImportError as e:
        error_msg = (
            "To use MistralOCR, please install the `mistral-ocr` extra: "
            "`pip install raglite[mistral-ocr]` or `uv add raglite[mistral-ocr]`."
        )
        raise ImportError(error_msg) from e

    api_key = _get_api_key(processor_config)
    return Mistral(api_key=api_key)


def _encode_document_base64(doc_path: Path) -> tuple[str, str]:
    """Encode a document as base64 with appropriate MIME type."""
    mime_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".avif": "image/avif",
        ".webp": "image/webp",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    mime_type = mime_types.get(doc_path.suffix.lower(), "application/octet-stream")
    data = base64.standard_b64encode(doc_path.read_bytes()).decode("utf-8")
    return data, mime_type


def _process_ocr_response(
    ocr_response: Any,
    *,
    include_image_descriptions: bool = True,
    exclude_image_types: frozenset[ImageType] | None = None,
) -> str:
    """Convert MistralOCR response to markdown string.

    When include_image_descriptions is True and bbox_annotation_format was used,
    image placeholders are replaced with their annotations.

    Parameters
    ----------
    ocr_response
        Response from Mistral OCR API.
    include_image_descriptions
        Whether to replace image placeholders with annotations.
    exclude_image_types
        Set of ImageType values to exclude from output.

    Returns
    -------
    str
        Document content as markdown.
    """
    exclude_image_types = exclude_image_types or frozenset()
    pages_md = []

    for page in ocr_response.pages:
        page_md = page.markdown

        if include_image_descriptions and page.images:
            for img in page.images:
                # Check if the image has an annotation (from bbox_annotation_format).
                annotation = getattr(img, "image_annotation", None)
                if annotation:
                    placeholder_pattern = rf"!\[[^\]]*\]\({re.escape(img.id)}\)"
                    # Parse annotation to check image type for filtering.
                    try:
                        parsed = ImageAnnotation.model_validate_json(annotation)
                        if parsed.image_type in exclude_image_types:
                            # Remove the image placeholder entirely.
                            page_md = re.sub(placeholder_pattern, "", page_md)
                            continue
                        replacement = f"[Image ({parsed.image_type.value}): {parsed.description}]"
                    except (ValueError, TypeError):
                        # If parsing fails, use raw annotation.
                        replacement = f"[Image: {annotation}]"
                    page_md = re.sub(placeholder_pattern, replacement, page_md)

        pages_md.append(page_md)

    return "\n\n".join(pages_md)


def mistral_ocr_to_markdown(doc_path: Path, *, processor_config: MistralOCRConfig) -> str:
    """Convert a document to markdown using Mistral OCR with vision annotations.

    Uses Mistral's bbox_annotation_format to automatically describe images and
    diagrams found in the document, making visual content searchable.

    Parameters
    ----------
    doc_path
        Path to the document file.
    processor_config
        MistralOCR processor configuration.

    Returns
    -------
    str
        Document content as GitHub Flavored Markdown with image descriptions.

    Raises
    ------
    ImportError
        If the mistralai package is not installed.
    ValueError
        If MISTRAL_API_KEY is not set and MistralOCRConfig.api_key is None.
    MistralOCRError
        If the OCR processing fails.
    """
    # Check if file extension is supported by MistralOCR.
    if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        # Fall back to default processor for unsupported types.
        from raglite._markdown import _default_document_to_markdown

        return _default_document_to_markdown(doc_path)

    try:
        from mistralai.extra import response_format_from_pydantic_model

        client = _get_mistral_client(processor_config)

        # Encode document as base64.
        data, mime_type = _encode_document_base64(doc_path)

        # Prepare document payload based on file type.
        if doc_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".avif", ".webp"}:
            document_payload = {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{data}",
            }
        else:
            # PDF, DOCX, PPTX.
            document_payload = {
                "type": "document_url",
                "document_url": f"data:{mime_type};base64,{data}",
            }

        # Build OCR request parameters.
        ocr_params: dict[str, Any] = {
            "model": "mistral-ocr-latest",
            "document": document_payload,
            "include_image_base64": False,  # We don't need base64, just annotations.
        }

        # Add bbox annotation format if image descriptions are enabled.
        if processor_config.include_image_descriptions:
            ocr_params["bbox_annotation_format"] = response_format_from_pydantic_model(
                ImageAnnotation
            )

        # Call MistralOCR API.
        ocr_response = client.ocr.process(**ocr_params)

        # Process response and replace image placeholders with annotations.
        return _process_ocr_response(
            ocr_response,
            include_image_descriptions=processor_config.include_image_descriptions,
            exclude_image_types=processor_config.exclude_image_types,
        )

    except ImportError:
        raise  # Re-raise import errors with the helpful message.
    except ValueError:
        raise  # Re-raise API key errors.
    except Exception as e:
        error_msg = f"MistralOCR failed to process {doc_path}: {e}"
        raise MistralOCRError(error_msg) from e
