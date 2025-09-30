"""Extract structured metadata from documents using LLMs."""

import json
import logging
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeVar, get_args, get_origin

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


def _unwrap_annotated(tp: Any) -> tuple[Any, str | None]:
    if get_origin(tp) is Annotated:
        base, *meta = get_args(tp)
        prompt = next((m for m in meta if isinstance(m, str)), None)
        return base, prompt
    return tp, None


def expand_document_metadata(  # noqa: C901, PLR0912, PLR0915
    documents: list[Document],
    metadata_fields: dict[str, Any],
    config: RAGLiteConfig,
    content_char_limit: int | None = None,
    source: str = "content",
    **kwargs: Any,
) -> Iterator[Document]:
    """
    Extract metadata from documents using an LLM.

    Parameters
    ----------
    documents
        Documents to enrich.
    metadata_fields
        A mapping ``{field_name: field_type}``.

        ``field_type`` may be a plain type (``str``, ``Literal[...]``, etc.) or
        ``typing.Annotated`` with a prompt string:

         from typing import Annotated, Literal
         metadata_fields = {
             "planet_mass": Annotated[float, "The mass of a planet"],
             "subject": Annotated[
                 Literal["Apple", "Meta", "Tesla"],
                 "The subject of the document"
             ],
             "participants": Annotated[
                 list[Literal["Alice", "Bob", "John", "Jane"]],
                 "People mentioned in the text"
             ],
         }
    config
        RAGLite configuration.
    content_char_limit
        If set, only the first ``content_char_limit`` characters of each
        document's content are sent to the LLM.
    source
        One of ``"content"`` (default) or the name of an existing metadata
        field to use as the extraction source.  If multiple sources are needed,
        call the function multiple times.
    **kwargs
        Passed through to ``litellm.batch_completion`` (e.g. ``max_workers``).

    Yields
    ------
    Document
        Documents with expanded ``metadata_``.
    """
    if not documents:
        return

    fields: dict[str, Any] = {}
    system_prompt = (
        "You are a metadata extractor for documents with perfect precision and recall.\n"
        "Extract the requested metadata from the provided source text.\n"
        "For constrained fields, only use the provided allowed values.\n"
        "For free-text fields, provide concise and accurate responses.\n"
        "Output valid JSON matching the required schema."
    )

    # Build dynamic Pydantic model for metadata validation and add field specs to system prompt.
    for field_name, declared_type in metadata_fields.items():
        field_type, prompt = _unwrap_annotated(declared_type)
        prompt = prompt or f"Extract '{field_name}'"

        # Determine if the field is single or multi-value,
        # and if it has any literal constraints.
        values = None
        is_multi_value = False
        origin = get_origin(field_type)
        if origin is list:
            args = get_args(field_type)
            if args and get_origin(args[0]) is Literal:
                values = list(get_args(args[0]))
                is_multi_value = True
            else:
                values = None
                is_multi_value = True
        elif origin is Literal:
            values = list(get_args(field_type))
            is_multi_value = False
        else:
            values = None
            is_multi_value = False

        # Build Pydantic field.
        if is_multi_value:  # Multi-value field (literal or free-text)
            fields[field_name] = (field_type, Field(default_factory=list))
        elif values:  # Single-value literal field
            fields[field_name] = (field_type, Field(default=values[0]))
        else:  # Single-value free-text fields: str, int, float, bool
            type_default_map = {
                str: "",
                int: 0,
                float: 0.0,
                bool: False,
            }
            if field_type in type_default_map:
                fields[field_name] = (field_type, Field(default=type_default_map[field_type]))
            else:
                msg = f"Unsupported field type for key '{field_name}': {field_type!r}"
                raise ValueError(msg)

        # Build system prompt for this field.
        source_label = "Content" if source == "content" else f"Metadata field '{source}'"
        field_prompt = f"**{field_name}**: {prompt}\n"
        field_prompt += f"  - Source: {source_label}\n"
        if values:  # Literal fields (single or multi-value)
            values_str = ", ".join(f'"{val}"' for val in values)
            field_prompt += f"  - Allowed values: {values_str}\n"
            if is_multi_value:
                field_prompt += "  - Select ALL that apply\n"
            else:
                field_prompt += "  - Select ONE that best fits\n"
        elif is_multi_value:  # Multi-value free-text field
            field_prompt += "  - Provide relevant values (can be multiple)\n"
        else:  # Single-value free-text field
            field_prompt += "  - Provide a single, concise response\n"
        system_prompt += field_prompt + "\n"

    model: type[BaseModel] = create_model("DocumentMetadata", __base__=BaseModel, **fields)

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

    # Add metadata and content for each document to messages.
    all_messages = []
    for doc in documents:
        source_text = doc.content or ""
        if content_char_limit is not None and len(source_text) > content_char_limit:
            source_text = source_text[:content_char_limit]
        user_prompt = "Metadata:\n" + json.dumps(dict(doc.metadata_)) + "\nContent:\n" + source_text
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
