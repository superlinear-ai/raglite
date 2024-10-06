"""Add support for llama-cpp-python models to LiteLLM."""

import logging
import warnings
from collections.abc import Callable, Iterator
from functools import cache
from typing import Any, ClassVar, cast

import httpx
import litellm
from litellm import (  # type: ignore[attr-defined]
    CustomLLM,
    GenericStreamingChunk,
    ModelResponse,
    convert_to_model_response_object,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from llama_cpp import (  # type: ignore[attr-defined]
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
    LlamaRAMCache,
)

# Reduce the logging level for LiteLLM and flashrank.
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("flashrank").setLevel(logging.WARNING)


class LlamaCppPythonLLM(CustomLLM):
    """A llama-cpp-python provider for LiteLLM.

    This provider enables using llama-cpp-python models with LiteLLM. The LiteLLM model
    specification is "llama-cpp-python/<hugging_face_repo_id>/<filename>@<n_ctx>", where n_ctx is
    an optional parameter that specifies the context size of the model. If n_ctx is not provided or
    if it's set to 0, the model's default context size is used.

    Example usage:

    ```python
    from litellm import completion

    response = completion(
        model="llama-cpp-python/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/*Q4_K_M.gguf@4092",
        messages=[{"role": "user", "content": "Hello world!"}],
        # stream=True
    )
    ```
    """

    # The set of supported OpenAI parameters is the intersection of [1] and [2]. Not included:
    # max_completion_tokens, stream_options, n, user, logprobs, top_logprobs, extra_headers.
    # [1] https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion
    # [2] https://docs.litellm.ai/docs/completion/input
    supported_openai_params: ClassVar[list[str]] = [
        "functions",  # Deprecated
        "function_call",  # Deprecated
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "stop",
        "seed",
        "response_format",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "tfs_z",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "logit_bias",
    ]

    @staticmethod
    @cache
    def llm(model: str, **kwargs: Any) -> Llama:
        # Drop the llama-cpp-python prefix from the model.
        repo_id_filename = model.replace("llama-cpp-python/", "")
        # Convert the LiteLLM model string to repo_id, filename, and n_ctx.
        repo_id, filename = repo_id_filename.rsplit("/", maxsplit=1)
        n_ctx = 0
        if len(filename_n_ctx := filename.rsplit("@", maxsplit=1)) == 2:  # noqa: PLR2004
            filename, n_ctx_str = filename_n_ctx
            n_ctx = int(n_ctx_str)
        # Load the LLM.
        with warnings.catch_warnings():  # Filter huggingface_hub warning about HF_TOKEN.
            warnings.filterwarnings("ignore", category=UserWarning)
            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                verbose=False,
                **kwargs,
            )
        # Enable caching.
        llm.set_cache(LlamaRAMCache())
        # Register the model info with LiteLLM.
        litellm.register_model(  # type: ignore[attr-defined]
            {
                model: {
                    "max_tokens": llm.n_ctx(),
                    "max_input_tokens": llm.n_ctx(),
                    "max_output_tokens": None,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "output_vector_size": llm.n_embd() if kwargs.get("embedding") else None,
                    "litellm_provider": "llama-cpp-python",
                    "mode": "embedding" if kwargs.get("embedding") else "completion",
                    "supported_openai_params": LlamaCppPythonLLM.supported_openai_params,
                    "supports_function_calling": True,
                    "supports_parallel_function_calling": True,
                    "supports_vision": False,
                }
            }
        )
        return llm

    def completion(  # noqa: PLR0913
        self,
        model: str,
        messages: list[ChatCompletionRequestMessage],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable,  # type: ignore[type-arg]
        encoding: str,
        api_key: str,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Callable | None = None,  # type: ignore[type-arg]
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable | None = None,  # type: ignore[type-arg]
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        llm = self.llm(model)
        llama_cpp_python_params = {
            k: v for k, v in optional_params.items() if k in self.supported_openai_params
        }
        response = cast(
            CreateChatCompletionResponse,
            llm.create_chat_completion(messages=messages, **llama_cpp_python_params),
        )
        litellm_model_response: ModelResponse = convert_to_model_response_object(
            response_object=response,
            model_response_object=model_response,
            response_type="completion",
            stream=False,
        )
        return litellm_model_response

    def streaming(  # noqa: PLR0913
        self,
        model: str,
        messages: list[ChatCompletionRequestMessage],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable,  # type: ignore[type-arg]
        encoding: str,
        api_key: str,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Callable | None = None,  # type: ignore[type-arg]
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable | None = None,  # type: ignore[type-arg]
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterator[GenericStreamingChunk]:
        llm = self.llm(model)
        llama_cpp_python_params = {
            k: v for k, v in optional_params.items() if k in self.supported_openai_params
        }
        stream = cast(
            Iterator[CreateChatCompletionStreamResponse],
            llm.create_chat_completion(messages=messages, **llama_cpp_python_params, stream=True),
        )
        for chunk in stream:
            choices = chunk.get("choices", [])
            for choice in choices:
                text = choice.get("delta", {}).get("content", None)
                finish_reason = choice.get("finish_reason")
                litellm_generic_streaming_chunk = GenericStreamingChunk(
                    text=text,  # type: ignore[typeddict-item]
                    is_finished=bool(finish_reason),
                    finish_reason=finish_reason,  # type: ignore[typeddict-item]
                    usage=None,
                    index=choice.get("index"),  # type: ignore[typeddict-item]
                    provider_specific_fields={
                        "id": chunk.get("id"),
                        "model": chunk.get("model"),
                        "created": chunk.get("created"),
                        "object": chunk.get("object"),
                    },
                )
                yield litellm_generic_streaming_chunk


# Register the LlamaCppPythonLLM provider.
if not any(provider["provider"] == "llama-cpp-python" for provider in litellm.custom_provider_map):
    litellm.custom_provider_map.append(
        {"provider": "llama-cpp-python", "custom_handler": LlamaCppPythonLLM()}
    )
    litellm.suppress_debug_info = True
