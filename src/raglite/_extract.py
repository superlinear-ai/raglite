"""Extract structured data from unstructured text with an LLM."""

from typing import Any, TypeVar

from litellm import completion, get_supported_openai_params  # type: ignore[attr-defined]
from pydantic import BaseModel, ValidationError

from raglite._config import RAGLiteConfig

T = TypeVar("T", bound=BaseModel)


def extract_with_llm(
    return_type: type[T],
    user_prompt: str | list[str],
    strict: bool = False,  # noqa: FBT001,FBT002
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
