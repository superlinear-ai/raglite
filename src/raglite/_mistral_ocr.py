"""MistralOCR document processor for RAGLite."""

import base64
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from raglite._config import MistralOCRConfig

logger = logging.getLogger(__name__)

# Single source of truth for supported extensions and their MIME types.
_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".avif": "image/avif",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
SUPPORTED_EXTENSIONS = frozenset(_MIME_TYPES)
_IMAGE_EXTENSIONS = frozenset(ext for ext, mime in _MIME_TYPES.items() if mime.startswith("image/"))


class MistralOCRError(Exception):
    """Error during MistralOCR processing."""


def _build_image_annotation_model(image_types: frozenset[str]) -> type[BaseModel]:
    """Build an ImageAnnotation Pydantic model with the given image types."""
    image_type_enum = Enum("ImageType", {t.upper(): t for t in sorted(image_types)}, type=str)  # type: ignore[misc]
    image_type_values = ", ".join(sorted(image_types))

    class ImageAnnotation(BaseModel):
        """Schema for vision-based image annotation."""

        image_type: image_type_enum = Field(  # type: ignore[valid-type]
            ...,
            description=f"The type of the image. Must be one of: {image_type_values}.",
        )
        description: str = Field(
            ...,
            description=(
                "A concise description of the image content. For diagrams and charts, "
                "describe what is being illustrated. For tables, summarize the data. "
                "For photos, describe the subject matter."
            ),
        )

    return ImageAnnotation


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


def _get_response_format_converter() -> Any:
    """Get the response_format_from_pydantic_model function from mistralai."""
    try:
        from mistralai.extra import response_format_from_pydantic_model
    except ImportError as e:
        error_msg = (
            "To use MistralOCR, please install the `mistral-ocr` extra: "
            "`uv add raglite[mistral-ocr]` or `pip install raglite[mistral-ocr]`."
        )
        raise ImportError(error_msg) from e
    return response_format_from_pydantic_model


def _encode_document_base64(doc_path: Path) -> tuple[str, str]:
    """Encode a document as base64 with appropriate MIME type."""
    mime_type = _MIME_TYPES.get(doc_path.suffix.lower(), "application/octet-stream")
    data = base64.standard_b64encode(doc_path.read_bytes()).decode("utf-8")
    return data, mime_type


def _process_ocr_response(
    ocr_response: Any,
    *,
    annotation_model: type[BaseModel],
    include_image_descriptions: bool = True,
    exclude_image_types: frozenset[str] | None = None,
) -> str:
    """Convert MistralOCR response to markdown string.

    When include_image_descriptions is True and bbox_annotation_format was used,
    image placeholders are replaced with their annotations.

    Parameters
    ----------
    ocr_response
        Response from Mistral OCR API.
    annotation_model
        The Pydantic model used to parse image annotations.
    include_image_descriptions
        Whether to replace image placeholders with annotations.
    exclude_image_types
        Set of image type strings to exclude from output.

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
                        parsed: Any = annotation_model.model_validate_json(annotation)
                        image_type = parsed.image_type.value
                        if image_type in exclude_image_types:
                            # Remove the image placeholder entirely.
                            page_md = re.sub(placeholder_pattern, "", page_md)
                            continue
                        replacement = f"[Image ({image_type}): {parsed.description}]"
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
    data, mime_type = _encode_document_base64(doc_path)

    if doc_path.suffix.lower() in _IMAGE_EXTENSIONS:
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
        "model": processor_config.model,
        "document": document_payload,
        "include_image_base64": False,  # We don't need base64, just annotations.
    }

    annotation_model = _build_image_annotation_model(processor_config.image_types)

    try:
        client = _get_mistral_client(processor_config)
        # Add bbox annotation format if image descriptions are enabled.
        if processor_config.include_image_descriptions:
            response_format_from_pydantic_model = _get_response_format_converter()
            ocr_params["bbox_annotation_format"] = response_format_from_pydantic_model(
                annotation_model
            )
        ocr_response = client.ocr.process(**ocr_params)
    except (ImportError, ValueError):
        raise
    except Exception as e:
        error_msg = f"MistralOCR failed to process {doc_path}: {e}"
        raise MistralOCRError(error_msg) from e

    # Process response and replace image placeholders with annotations.
    return _process_ocr_response(
        ocr_response,
        annotation_model=annotation_model,
        include_image_descriptions=processor_config.include_image_descriptions,
        exclude_image_types=processor_config.exclude_image_types,
    )
