"""Extract structured metadata from documents using LLMs."""

import logging
from collections.abc import Iterator
from typing import Any, Literal, TypeVar, get_args, get_origin

from litellm import (  # type: ignore[attr-defined]
    batch_completion,
    completion,
    get_supported_openai_params,
)
from pydantic import BaseModel, Field, ValidationError, create_model

from raglite import Document
from raglite._config import RAGLiteConfig

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


def extract_with_llm(
    return_type: type[T],
    user_prompt: str | list[str],
    strict: bool = False,  # noqa: FBT001, FBT002
    config: RAGLiteConfig | None = None,
    **kwargs: Any,
) -> T:
    """Extract structured data from unstructured text with an LLM.

    This function expects a `return_type.system_prompt: ClassVar[str]` that contains the system
    prompt to use. Example:

        from typing import ClassVar
        from pydantic import BaseModel, Field

        class MyNameResponse(BaseModel):
            my_name: str = Field(..., description="The user's name.")
            system_prompt: ClassVar[str] = "The system prompt to use (excluded from JSON schema)."

        my_name_response = extract_with_llm(MyNameResponse, "My name is Thomas A. Anderson.")
    """
    # Load the default config if not provided.
    config = config or RAGLiteConfig()
    # Check if the LLM supports the response format.
    llm_supports_response_format = "response_format" in (
        get_supported_openai_params(model=config.llm) or []
    )
    # Update the system prompt with the JSON schema of the return type to help the LLM.
    system_prompt = getattr(return_type, "system_prompt", "").strip()
    if not llm_supports_response_format or config.llm.startswith("llama-cpp-python"):
        system_prompt += f"\n\nFormat your response according to this JSON schema:\n{return_type.model_json_schema()!s}"
    # Constrain the response format to the JSON schema if it's supported by the LLM [1]. Strict mode
    # is disabled by default because it only supports a subset of JSON schema features [2].
    # [1] https://docs.litellm.ai/docs/completion/json_mode
    # [2] https://platform.openai.com/docs/guides/structured-outputs#some-type-specific-keywords-are-not-yet-supported
    # TODO: Fall back to {"type": "json_object"} if JSON schema is not supported by the LLM.
    response_format: dict[str, Any] | None = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": return_type.__name__,
                "description": return_type.__doc__ or "",
                "schema": return_type.model_json_schema(),
                "strict": strict,
            },
        }
        if llm_supports_response_format
        else None
    )
    # Concatenate the user prompt if it is a list of strings.
    if isinstance(user_prompt, list):
        user_prompt = "\n\n".join(
            f'<context index="{i + 1}">\n{chunk.strip()}\n</context>'
            for i, chunk in enumerate(user_prompt)
        )
    # Extract structured data from the unstructured input.
    for _ in range(config.llm_max_tries):
        response = completion(
            model=config.llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            **kwargs,
        )
        try:
            instance = return_type.model_validate_json(response["choices"][0]["message"]["content"])
        except (KeyError, ValueError, ValidationError) as e:
            # Malformed response, not a JSON string, or not a valid instance of the return type.
            last_exception = e
            continue
        else:
            break
    else:
        error_message = f"Failed to extract {return_type} from input {user_prompt}."
        raise ValueError(error_message) from last_exception
    return instance


def _extract_literal_values(field_type: Any) -> tuple[list[str] | None, bool]:
    """Extract literal values and determine if it's a multi-value field."""
    if get_origin(field_type) is list:
        args = get_args(field_type)
        if args and get_origin(args[0]) is Literal:  # Multi-value literal field.
            return list(get_args(args[0])), True
        # Multi-value free-text field like list[str].
        return None, True
    if get_origin(field_type) is Literal:  # Single-value literal field.
        return list(get_args(field_type)), False

    return None, False  # Single-value free-text field.


def _build_field_prompt(field_spec: dict[str, Any], doc: Document) -> str:
    """Build prompt for a single metadata field."""
    key = field_spec["key"]
    field_type = field_spec["type"]
    prompt = field_spec["prompt"]
    source = field_spec.get("source", "content")
    max_chars = field_spec.get("max_chars")

    values, is_multi_value = _extract_literal_values(field_type)

    if source == "content":
        source_text = doc.content
        source_label = "document content"
        if source_text is None or source_text.strip() == "":
            msg = "Document content is empty; cannot extract metadata from content."
            raise ValueError(msg)
    else:
        source_text = doc.metadata_.get(source)
        source_label = f"metadata field '{source}'"
        if source_text is None:
            msg = f"Metadata field '{source}' not found in document metadata."
            raise ValueError(msg)

    # Apply character limit if specified
    if max_chars is not None and isinstance(source_text, str) and len(source_text) > max_chars:
        source_text = source_text[:max_chars]
        source_label += f" (truncated to {max_chars} chars)"

    field_prompt = f"**{key}**: {prompt}\n"
    field_prompt += f"  - Source: {source_label}\n"
    field_prompt += f"  - Text to analyze: {source_text}\n"

    # TODO: Determine if we want to add this also when supporting response_format
    if values:  # Literal fields (single or multi-value)
        values_str = ", ".join(f'"{val}"' for val in values)
        if is_multi_value:
            field_prompt += f"  - Allowed values: {values_str}\n"
            field_prompt += "  - Select ALL that apply\n"
        else:
            field_prompt += f"  - Allowed values: {values_str}\n"
            field_prompt += "  - Select ONE that best fits\n"
    elif is_multi_value:  # Multi-value free-text field
        field_prompt += "  - Provide relevant values (can be multiple)\n"
    else:  # Single-value free-text field
        field_prompt += "  - Provide a single, concise response\n"

    return field_prompt + "\n"


def expand_document_metadata(  # noqa: C901, PLR0912, PLR0915
    documents: list[Document],
    metadata_fields: list[dict[str, Any]],
    config: RAGLiteConfig,
    **kwargs: Any,
) -> Iterator[Document]:
    """
    Extract metadata from documents using LLM with configurable fields and constraints.

    Parameters
    ----------
    documents : list[Document]
        List of documents to process
    metadata_fields : list[dict[str, Any]]
        List of metadata field specifications. Each dict should contain:
        - key (str): The metadata field name
        - type (Any): The exact type for the field:
            (a.) Literal["A","B"] for single-choice,
            (b.) list[Literal["X","Y","Z"]] for multi-choice,
            (c.) str, int, float, bool for single-value free-text fields, and
            (d.) list[str], list[int], list[float], list[bool] for multi-value free-text fields
        - prompt (str): Instruction to guide the LLM in extracting the field
        - source (str, optional): Source for extraction - either "content" (default)
          or metadata field key
        - max_chars (int, optional): Maximum number of characters to consider from the source text
    config : RAGLiteConfig
        RAGLite configuration
    **kwargs : Any
        Additional keyword arguments passed to `batch_completion` (e.g., max_workers, threading)

    Returns
    -------
    Iterator[Document]
        Documents with expanded metadata

    Examples
    --------
    Extract document metadata from content and existing metadata before inserting into database:

        from raglite import Document, RAGLiteConfig, insert_documents, expand_document_metadata
        documents = [
            Document.from_path(path/to/document1.md, author = "John Doe"),
            Document.from_path(path/to/document2.md, author = "Jane Smith"),
            Document.from_path(path/to/document3.pdf, author = "Alice Johnson"),
        ]
        metadata_fields = [
            {
                "key": "document_type",
                "type": Literal["research-paper", "tutorial", "documentation"],
                "prompt": "What type of document is this?",
                "source": "content"  # Extract from document content
            },
            {
                "key": "author_affiliation",
                "type": Literal["academic", "industry", "government"],
                "prompt": "What type of affiliation does this author likely have?",
                "source": "author"  # Extract from existing 'author' metadata field
            },
            {
                "key": "topics",
                "type": list[Literal["AI", "ML", "NLP", "computer-vision", "robotics"]],
                "prompt": "What topics does this document cover?",
                "source": "content"  # Multi-choice field
            },
            {
                "key": "summary",
                "type": str,
                "prompt": "Provide a brief summary of this document",
                "source": "content",  # Extract from document content (default)
                "max_chars": 2000  # Only analyze first 2000 characters
            }
        ]

        # Expand metadata before inserting to database
        expanded_docs = list(
            expand_document_metadata(documents, metadata_fields, raglite_config, max_workers=4)
        )

        # Now insert to database
        insert_documents(expanded_docs)
    """
    if not documents:
        return

    # Create dynamic Pydantic model for metadata validation.
    fields: dict[str, Any] = {}
    for field_spec in metadata_fields:
        key = field_spec["key"]
        field_type = field_spec["type"]

        values, is_multi_value = _extract_literal_values(field_type)

        if is_multi_value:  # Multi-value field (literal or free-text)
            fields[key] = (field_type, Field(default_factory=list))
        elif values:  # Single-value literal field
            fields[key] = (field_type, Field(default=values[0]))
        else:  # Single-value free-text fields: str, int, float, bool
            type_default_map = {
                str: "",
                int: 0,
                float: 0.0,
                bool: False,
            }
            if field_type in type_default_map:
                fields[key] = (field_type, Field(default=type_default_map[field_type]))
            else:
                msg = f"Unsupported field type for key '{key}': {field_type!r}"
                raise ValueError(msg)

    model: type[BaseModel] = create_model("DocumentMetadata", __base__=BaseModel, **fields)

    # Base system prompt.
    system_prompt = (
        "You are a metadata extractor for documents with perfect precision and recall.\n"
        "Extract the requested metadata from the provided source text.\n"
        "For constrained fields, only use the provided allowed values.\n"
        "For free-text fields, provide concise and accurate responses.\n"
        "Output valid JSON matching the required schema."
    )

    # Construct response format if supported by the LLM.
    supports_rf = "response_format" in (get_supported_openai_params(model=config.llm) or [])
    response_format: dict[str, Any] | None = None

    if supports_rf:
        schema = model.model_json_schema()
        # OpenAI-specific for strict mode:
        # - additionalProperties must be false [1].
        # - all fields must be in the required array [2].
        # [1] https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
        # [2] https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "description": "Document metadata extraction",
                "schema": schema,
                "strict": True,
            },
        }
    else:
        system_prompt += (
            f"\n\nFormat your response according to this JSON schema:\n{model.model_json_schema()}"
        )
        response_format = None

    # Build messages for batch processing.
    all_messages = []
    for doc in documents:
        user_prompt = "Extract the following metadata:\n\n"
        for field_spec in metadata_fields:
            user_prompt += _build_field_prompt(field_spec, doc)
        all_messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    # Process requests using batch_completion.
    if response_format:
        responses = batch_completion(
            model=config.llm,
            messages=all_messages,
            response_format=response_format,
            **kwargs,
        )
    else:
        responses = batch_completion(model=config.llm, messages=all_messages, **kwargs)

    # Yield documents with expanded metadata.
    for doc, response in zip(documents, responses, strict=True):
        success = False
        doc_name = getattr(doc, "filename", doc.id)
        if isinstance(response, Exception):
            data = {}
            logger.warning(
                "[RAGLite] Metadata extraction failed for document: %s (Exception: %r)",
                doc_name,
                response,
            )
        else:
            try:
                content = response["choices"][0]["message"]["content"]
                data = model.model_validate_json(content).model_dump(exclude_none=True)
                success = True
            except (KeyError, ValueError, ValidationError) as e:
                data = {}
                logger.warning(
                    "[RAGLite] Metadata extraction failed for document: %s (Error: %r)", doc_name, e
                )
        if success:
            logger.info("[RAGLite] Metadata extraction succeeded for document: %s", doc_name)

        yield Document(
            id=doc.id,
            filename=doc.filename,
            url=doc.url,
            metadata_={**dict(doc.metadata_), **data},
            content=doc.content,
        )
