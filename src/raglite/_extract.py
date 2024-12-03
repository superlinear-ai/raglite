"""Extract structured data from unstructured text with an LLM."""

from typing import Any, TypeVar

import litellm
from litellm import completion, get_supported_openai_params  # type: ignore[attr-defined]
from pydantic import BaseModel, ValidationError

from raglite._config import RAGLiteConfig

T = TypeVar("T", bound=BaseModel)


def extract_with_llm(
    return_type: type[T],
    user_prompt: str | list[str],
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
    # Update the system prompt with the JSON schema of the return type to help the LLM.
    system_prompt = "\n".join(
        (
            return_type.system_prompt.strip(),  # type: ignore[attr-defined]
            "Format your response according to this JSON schema:",
            str(return_type.model_json_schema()),
        )
    )
    # Constrain the reponse format to the JSON schema if it's supported by the LLM [1].
    # [1] https://docs.litellm.ai/docs/completion/json_mode
    # TODO: Fall back to {"type": "json_object"} if JSON schema is not supported by the LLM.
    llm_provider = "llama-cpp-python" if config.embedder.startswith("llama-cpp") else None
    response_format: dict[str, Any] | None = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": return_type.__name__,
                "description": return_type.__doc__ or "",
                "schema": return_type.model_json_schema(),
            },
        }
        if "response_format"
        in (get_supported_openai_params(model=config.llm, custom_llm_provider=llm_provider) or [])
        else None
    )
    # Concatenate the user prompt if it is a list of strings.
    if isinstance(user_prompt, list):
        user_prompt = "\n\n".join(
            f'<context index="{i + 1}">\n{chunk.strip()}\n</context>'
            for i, chunk in enumerate(user_prompt)
        )
    # Enable JSON schema validation.
    enable_json_schema_validation = litellm.enable_json_schema_validation
    litellm.enable_json_schema_validation = True
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
    # Restore the previous JSON schema validation setting.
    litellm.enable_json_schema_validation = enable_json_schema_validation
    return instance
