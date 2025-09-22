"""Extract structured metadata from documents using LLMs."""

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
    """Extract literal values and determine if it's a multi-choice field."""
    # Check if it's a list type
    if get_origin(field_type) is list:
        # Extract the inner type from list
        args = get_args(field_type)
        if args and get_origin(args[0]) is Literal:
            # Multi-choice literal field
            return list(get_args(args[0])), True
    elif get_origin(field_type) is Literal:
        # Single-choice literal field
        return list(get_args(field_type)), False

    # Not a literal type
    return None, False


def _build_field_prompt(field_spec: dict[str, Any], doc: Document) -> str:
    """Build prompt for a single metadata field."""
    key = field_spec["key"]
    field_type = field_spec["type"]
    prompt = field_spec["prompt"]
    source = field_spec.get("source", "content")

    # Extract literal values and determine if multi-choice
    values, allow_multiple = _extract_literal_values(field_type)

    # Get source text based on source specification
    if source == "content":
        source_text = doc.content or ""
        source_label = "document content"
    else:
        # Extract from existing metadata field
        source_text = str(doc.metadata_.get(source, ""))
        source_label = f"metadata field '{source}'"

    field_prompt = f"**{key}**: {prompt}\n"
    field_prompt += f"  - Source: {source_label}\n"
    field_prompt += f"  - Text to analyze: {source_text}\n"

    if values:
        values_str = ", ".join(f'"{val}"' for val in values)
        if allow_multiple:
            field_prompt += f"  - Allowed values: {values_str}\n"
            field_prompt += "  - Select ALL that apply\n"
        else:
            field_prompt += f"  - Allowed values: {values_str}\n"
            field_prompt += "  - Select ONE that best fits\n"
    elif allow_multiple:
        field_prompt += "  - Provide relevant values (can be multiple)\n"
    else:
        field_prompt += "  - Provide a single, concise response\n"

    return field_prompt + "\n"


def expand_document_metadata(  # noqa: C901, PLR0912
    documents: list[Document],
    metadata_fields: list[dict[str, Any]],
    config: RAGLiteConfig,
    *,
    max_concurrent_requests: int | None = None,
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
        - type (Any): The exact type for the field, e.g., Literal["A","B"] for single-choice
          or list[Literal["X","Y","Z"]] for multi-choice
        - prompt (str): Instruction for the LLM on what to extract
        - source (str, optional): Source for extraction - either "content" (default)
          or metadata field key
    config : RAGLiteConfig
        RAGLite configuration
    max_concurrent_requests : int, optional
        Maximum number of concurrent LLM requests


    Returns
    -------
    Iterator[Document]
        Documents with expanded metadata

    Examples
    --------
    Extract document metadata from content and existing metadata before inserting into database:

        documents = [
            Document(
                content="This is a research paper about machine learning...",
                metadata_={"author": "John Doe", "year": "2024"}
            ),
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
                "source": "content"  # Extract from document content (default)
            }
        ]

        # Expand metadata before inserting to database
        expanded_docs = list(expand_document_metadata(documents, metadata_fields))

        # Now insert to database
        db.insert(expanded_docs)
    """
    if not documents:
        yield from ()

    # Build field specifications for Pydantic model
    fields: dict[str, Any] = {}
    for field_spec in metadata_fields:
        key = field_spec["key"]
        field_type = field_spec["type"]

        # Check if it's a list type
        if get_origin(field_type) is list:
            # Multi-choice field: use as-is with list default
            fields[key] = (field_type, Field(default_factory=list))
        else:
            # For non-list fields, we need to handle defaults
            # Extract literal values to get a valid default
            values, _ = _extract_literal_values(field_type)
            if values:
                # Use the first literal value as default
                fields[key] = (field_type, Field(default=values[0]))
            else:
                # For non-literal types like str, use empty string
                fields[key] = (field_type, Field(default=""))

    model = create_model("DocumentMetadata", __base__=BaseModel, **fields)

    # Build system prompt
    system_prompt = (
        "You are a precise metadata extractor for documents.\n"
        "Extract the requested metadata from the provided source text.\n"
        "For constrained fields, only use the provided allowed values.\n"
        "For free-text fields, provide concise and accurate responses.\n"
        "Output valid JSON matching the required schema."
    )

    # Check if LLM supports structured output
    supports_rf = "response_format" in (get_supported_openai_params(model=config.llm) or [])
    response_format: dict[str, Any] | None = None

    if supports_rf:
        schema = model.model_json_schema()
        # OpenAI requires additionalProperties to be false for strict mode
        schema["additionalProperties"] = False

        # For strict mode, all properties must be in the required array
        # Get all property names and add them to required
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

    # Build requests for batch processing
    reqs: list[dict[str, Any]] = []
    for doc in documents:
        # Build user prompt with field-specific instructions
        user_prompt = "Please extract the following metadata:\n\n"

        for field_spec in metadata_fields:
            user_prompt += _build_field_prompt(field_spec, doc)

        body: dict[str, Any] = {
            "model": config.llm,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format is not None:
            body["response_format"] = response_format
        reqs.append(body)

    # Process requests using batch_completion with correct signature
    all_messages = [req["messages"] for req in reqs]
    response_format = reqs[0].get("response_format") if reqs else None

    if response_format:
        responses = batch_completion(
            model=config.llm,
            messages=all_messages,
            max_workers=max_concurrent_requests,
            response_format=response_format,
        )
    else:
        responses = batch_completion(
            model=config.llm, messages=all_messages, max_workers=max_concurrent_requests
        )

    # Yield documents with expanded metadata
    for doc, response in zip(documents, responses, strict=True):
        if isinstance(response, Exception):
            data = {}
        else:
            try:
                content = response["choices"][0]["message"]["content"]
                data = model.model_validate_json(content).model_dump(exclude_none=True)
            except (KeyError, ValueError, ValidationError):
                data = {}

        yield Document(
            id=doc.id,
            filename=doc.filename,
            url=doc.url,
            metadata_={**dict(doc.metadata_), **data},
            content=doc.content,
        )
