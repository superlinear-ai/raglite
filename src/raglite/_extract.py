"""Extract structured data from unstructured text with an LLM."""

from typing import Any, TypeVar

from litellm import completion, get_supported_openai_params  # type: ignore[attr-defined]
from pydantic import BaseModel, ValidationError

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


def extract_metadata(  # noqa: PLR0913
    text: str,
    prompt: str,
    key: str,
    allowed_values: list[str] | None = None,
    *,
    allow_multiple: bool = True,
    config: RAGLiteConfig | None = None,
    **kwargs: Any,
) -> dict[str, str | list[str]]:
    """Extract metadata from text using an LLM.

    Parameters
    ----------
    text : str
        The input text to analyze
    prompt : str
        Instruction for the LLM on what metadata to extract
    key : str
        The metadata field name.
    allowed_values : list[str], optional
        List of valid values that can be returned. If None, free-text mode is used.
    allow_multiple : bool, optional
        Whether to allow multiple values (default: True)
    config : RAGLiteConfig, optional
        RAGLite configuration
    **kwargs : Any
        Additional arguments passed to the LLM

    Returns
    -------
    dict[str, str | list[str]]
        Dictionary with a single key-value pair containing the extracted metadata

    Examples
    --------
    >>> metadata = extract_metadata(
    ...     text="This is a research paper about machine learning algorithms.",
    ...     prompt="What type of document is this?",
    ...     key="document_type",
    ...     allowed_values=["research", "tutorial", "documentation"],
    ...     allow_multiple=False,
    ... )
    >>> print(metadata)
    {"document_type": "research"}

    >>> metadata = extract_metadata(
    ...     text="This is a research paper about machine learning algorithms.",
    ...     prompt="Summarize the main topic",
    ...     key="summary",
    ... )
    >>> print(metadata)
    {"summary": "Machine learning algorithm implementation"}
    """
    config = config or RAGLiteConfig()

    # Build the user prompt
    user_prompt = f"{prompt}\n\nText to analyze:\n{text}"

    if allowed_values:
        # Constrained mode
        values_str = ", ".join(f'"{val}"' for val in allowed_values)
        if allow_multiple:
            instruction = "Select ALL values that apply from the list. Return them comma-separated."
        else:
            instruction = (
                "Select ONLY ONE value that best fits from the list. Return just that single value."
            )
        user_prompt += f"\n\nAllowed values: {values_str}\nInstructions: {instruction}"
    else:
        # Free-text mode
        if allow_multiple:
            instruction = "Provide multiple relevant values or phrases. If asked, provide them as comma-separated values."
        else:
            instruction = "Provide a single, concise answer."
        user_prompt += f"\n\nInstructions: {instruction}"

    # Call LLM
    response = completion(
        model=config.llm,
        messages=[
            {
                "role": "system",
                "content": "Extract the requested metadata from the text. Be concise and precise.",
            },
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )

    result = response["choices"][0]["message"]["content"].strip()

    # Post-process result based on mode
    if allowed_values:
        # Constrained mode: filter and validate
        extracted_values = [val.strip().strip("\"'") for val in result.split(",")]
        valid_values = [val for val in extracted_values if val in allowed_values]

        final_result = valid_values if allow_multiple else (valid_values[0] if valid_values else "")
    elif allow_multiple:
        # Free-text mode: multiple values
        values = [val.strip() for val in result.split(",")]
        final_result = values
    else:
        # Free-text mode: single value
        final_result = result

    return {key: final_result}
