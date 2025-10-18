"""Extract structured metadata from documents using LLMs."""

import warnings
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeVar

from litellm import (  # type: ignore[attr-defined]
    batch_completion,
    completion,
    get_supported_openai_params,
)
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from raglite import Document
from raglite._config import RAGLiteConfig

T = TypeVar("T", bound=BaseModel)


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
        system_prompt += f"\n\nFormat your response according to this JSON schema:\n{return_type.model_json_schema()}"
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


def expand_document_metadata(  # noqa: PLR0913
    documents: Iterable[Document],
    metadata_fields: Mapping[str, type[Any]],
    *,
    max_context_size: int | None = None,
    source: str = "content",
    strict: bool = False,
    config: RAGLiteConfig | None = None,
    **kwargs: Any,
) -> Iterator[Document]:
    """Extract metadata for each document with help from an LLM.

    Parameters
    ----------
    documents
        Documents to enrich. When ``source`` is ``"content"``, each document must expose its
        ``content``.
    metadata_fields
        Mapping from metadata field name to its expected type and Field configuration. Each value
        must be an ``Annotated`` declaration combining the type with ``Field(..., description=...)``
        (or ``Field(default=None, ...)`` for optional fields).

        Example
        -------
        >>> from typing import Annotated, Literal, Optional
        >>> from pydantic import Field
        >>> metadata_fields = {
        ...     "title": Annotated[str, Field(..., description="Document title.")],
        ...     "pages": Annotated[int, Field(..., description="Total page count.")],
        ...     "rating": Annotated[float, Field(..., description="Average review score.")],
        ...     "reviewed": Annotated[bool, Field(..., description="Whether peer reviewed.")],
        ...     "category": Annotated[
        ...         Literal["Planet", "Moon"],
        ...         Field(..., description="Primary classification."),
        ...     ],
        ...     "tags": Annotated[
        ...         list[Literal["Exploration", "Geology"]],
        ...         Field(..., description="Relevant themes."),
        ...     ],
        ...     "participants": Annotated[
        ...         list[str],
        ...         Field(..., description="Contributors."),
        ...     ],
        ...     "editor": Annotated[
        ...         Optional[str],
        ...         Field(default=None, description="Editor name (if available)"),
        ...     ],
        ... }

    max_context_size
        Maximum number of characters copied from the source text before invoking the LLM.
    source
        ``"content"`` (default) to rely on ``Document.content`` or the name of an existing metadata
        key that provides the extraction source.
    strict
        Whether to enforce strict adherence to the JSON schema when extracting metadata.
    config
        RAGLite configuration. Defaults to ``RAGLiteConfig()`` when omitted.
    **kwargs
        Additional keyword arguments forwarded to ``litellm.batch_completion``.

    Yields
    ------
    Iterator[Document]
        Documents whose metadata is expanded with the extracted metadata fields.
    """
    # Return early if there are no documents.
    if not documents:
        return
    # Load the default config if not provided.
    config = config or RAGLiteConfig()
    # Build a Pydantic model for the metadata fields.
    metadata_model = create_model(  # type: ignore[call-overload]
        "DocumentMetadata",
        __config__=ConfigDict(extra="forbid"),
        **metadata_fields,
    )
    # Prepare the system prompt and response format.
    # TODO: Fall back to {"type": "json_object"} if JSON schema is not supported by the LLM.
    llm_supports_response_format = "response_format" in (
        get_supported_openai_params(model=config.llm) or []
    )
    system_prompt = (
        "You are a metadata extractor with perfect precision and recall.\n"
        "Extract the requested metadata from the provided source text.\n"
        "For constrained fields, only use the allowed values.\n"
        "For free-text fields, provide concise and accurate responses.\n"
        "Output valid JSON that matches the schema."
    )
    if not llm_supports_response_format or config.llm.startswith("llama-cpp-python"):
        system_prompt += f"\n\nFormat your response according to this JSON schema:\n{metadata_model.model_json_schema()}"
    response_format = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": metadata_model.__name__,
                "description": metadata_model.__doc__ or "",
                "schema": metadata_model.model_json_schema(),
                "strict": strict,
            },
        }
        if llm_supports_response_format
        else None
    )
    # Batch process the documents to extract metadata.
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{document.front_matter}\n\n{document.content}".strip()
                    if source == "content"
                    else f"{source}: {document.metadata_.get(source, '')}"
                )[:max_context_size],
            },
        ]
        for document in documents
    ]
    responses = batch_completion(
        model=config.llm, messages=messages, response_format=response_format, **kwargs
    )
    # Extract the metadata from the responses.
    for document, response in zip(documents, responses, strict=True):
        extracted_metadata: dict[str, Any]
        if isinstance(response, Exception):
            extracted_metadata = {}
            warnings.warn(f"Metadata extraction failed for {document!r}: {response}", stacklevel=2)
        else:
            try:
                content = response["choices"][0]["message"]["content"]
                extracted_metadata = metadata_model.model_validate_json(content).model_dump(
                    exclude_unset=True
                )
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                extracted_metadata = {}
                warnings.warn(f"Metadata extraction failed for {document!r}: {exc}", stacklevel=2)
        yield Document(
            id=document.id,
            filename=document.filename,
            url=document.url,
            metadata_={**document.metadata_, **extracted_metadata},
            content=document.content,
        )
