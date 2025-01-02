"""Add support for llama-cpp-python models to LiteLLM."""

import asyncio
import contextlib
import logging
import os
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from functools import cache
from io import StringIO
from typing import Any, ClassVar, cast

import httpx
import litellm
from litellm import (  # type: ignore[attr-defined]
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    CustomLLM,
    GenericStreamingChunk,
    ModelResponse,
    convert_to_model_response_object,
    get_model_info,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import custom_llm_setup
from llama_cpp import (  # type: ignore[attr-defined]
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
    LlamaRAMCache,
)

from raglite._chatml_function_calling import chatml_function_calling_with_streaming
from raglite._config import RAGLiteConfig

# Reduce the logging level for LiteLLM, flashrank, and httpx.
litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "WARNING"
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("flashrank").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


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

    # Create a lock to prevent concurrent access to llama-cpp-python models.
    streaming_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

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
        with (
            contextlib.redirect_stderr(StringIO()),  # Filter spurious llama.cpp output.
            warnings.catch_warnings(),  # Filter huggingface_hub warning about HF_TOKEN.
        ):
            warnings.filterwarnings("ignore", category=UserWarning)
            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                verbose=False,
                # Enable function calling with streaming.
                chat_handler=chatml_function_calling_with_streaming,
                # Workaround to enable long context embedding models [1].
                # [1] https://github.com/abetlen/llama-cpp-python/issues/1762
                n_batch=n_ctx if n_ctx > 0 else 1024,
                n_ubatch=n_ctx if n_ctx > 0 else 1024,
                **kwargs,
            )
        # Enable caching.
        llm.set_cache(LlamaRAMCache())
        # Register the model info with LiteLLM.
        model_info = {
            repo_id_filename: {
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
        litellm.register_model(model_info)  # type: ignore[attr-defined]
        return llm

    def _translate_openai_params(self, optional_params: dict[str, Any]) -> dict[str, Any]:
        # Filter out unsupported OpenAI parameters.
        llama_cpp_python_params = {
            k: v for k, v in optional_params.items() if k in self.supported_openai_params
        }
        # Translate OpenAI's response_format [1] to llama-cpp-python's response_format [2].
        # [1] https://platform.openai.com/docs/guides/structured-outputs
        # [2] https://github.com/abetlen/llama-cpp-python#json-schema-mode
        if (
            "response_format" in llama_cpp_python_params
            and "json_schema" in llama_cpp_python_params["response_format"]
        ):
            llama_cpp_python_params["response_format"] = {
                "type": "json_object",
                "schema": llama_cpp_python_params["response_format"]["json_schema"]["schema"],
            }
        return llama_cpp_python_params

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
        llama_cpp_python_params = self._translate_openai_params(optional_params)
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
        llama_cpp_python_params = self._translate_openai_params(optional_params)
        stream = cast(
            Iterator[CreateChatCompletionStreamResponse],
            llm.create_chat_completion(messages=messages, **llama_cpp_python_params, stream=True),
        )
        for chunk in stream:
            choices = chunk.get("choices")
            if not choices:
                continue
            text = choices[0].get("delta", {}).get("content", None)
            tool_calls = choices[0].get("delta", {}).get("tool_calls", None)
            tool_use = (
                ChatCompletionToolCallChunk(
                    id=tool_calls[0]["id"],  # type: ignore[index]
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=tool_calls[0]["function"]["name"],  # type: ignore[index]
                        arguments=tool_calls[0]["function"]["arguments"],  # type: ignore[index]
                    ),
                    index=tool_calls[0]["index"],  # type: ignore[index]
                )
                if tool_calls
                else None
            )
            finish_reason = choices[0].get("finish_reason")
            litellm_generic_streaming_chunk = GenericStreamingChunk(
                text=text,  # type: ignore[typeddict-item]
                tool_use=tool_use,
                is_finished=bool(finish_reason),
                finish_reason=finish_reason,  # type: ignore[typeddict-item]
                usage=None,
                index=choices[0].get("index"),  # type: ignore[typeddict-item]
                provider_specific_fields={
                    "id": chunk.get("id"),
                    "model": chunk.get("model"),
                    "created": chunk.get("created"),
                    "object": chunk.get("object"),
                },
            )
            yield litellm_generic_streaming_chunk

    async def astreaming(  # type: ignore[misc,override]  # noqa: PLR0913
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
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        # Start a synchronous stream.
        stream = self.streaming(
            model,
            messages,
            api_base,
            custom_prompt_dict,
            model_response,
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            optional_params,
            acompletion,
            litellm_params,
            logger_fn,
            headers,
            timeout,
        )
        await asyncio.sleep(0)  # Yield control to the event loop after initialising the context.
        # Wrap the synchronous stream in an asynchronous stream.
        async with LlamaCppPythonLLM.streaming_lock:
            for litellm_generic_streaming_chunk in stream:
                yield litellm_generic_streaming_chunk
                await asyncio.sleep(0)  # Yield control to the event loop after each token.


# Register the LlamaCppPythonLLM provider.
if not any(provider["provider"] == "llama-cpp-python" for provider in litellm.custom_provider_map):
    litellm.custom_provider_map.append(
        {"provider": "llama-cpp-python", "custom_handler": LlamaCppPythonLLM()}
    )
    custom_llm_setup()  # type: ignore[no-untyped-call]


@cache
def get_context_size(config: RAGLiteConfig, *, fallback: int = 2048) -> int:
    """Get the context size for the configured LLM."""
    # If the user has configured a llama-cpp-python model, we ensure that LiteLLM's model info is up
    # to date by loading that LLM.
    if config.llm.startswith("llama-cpp-python"):
        _ = LlamaCppPythonLLM.llm(config.llm)
    # Attempt to read the context size from LiteLLM's model info.
    model_info = get_model_info(config.llm)
    max_tokens = model_info.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        return max_tokens
    # Fall back to a default context size if the model info is not available.
    if fallback > 0:
        warnings.warn(
            f"Could not determine the context size of {config.llm} from LiteLLM's model_info, using {fallback}.",
            stacklevel=2,
        )
        return 2048
    error_message = f"Could not determine the context size of {config.llm}."
    raise ValueError(error_message)


@cache
def get_embedding_dim(config: RAGLiteConfig, *, fallback: bool = True) -> int:
    """Get the embedding dimension for the configured embedder."""
    # If the user has configured a llama-cpp-python model, we ensure that LiteLLM's model info is up
    # to date by loading that LLM.
    if config.embedder.startswith("llama-cpp-python"):
        _ = LlamaCppPythonLLM.llm(config.embedder, embedding=True)
    # Attempt to read the embedding dimension from LiteLLM's model info.
    model_info = get_model_info(config.embedder)
    embedding_dim = model_info.get("output_vector_size")
    if isinstance(embedding_dim, int) and embedding_dim > 0:
        return embedding_dim
    # If that fails, fall back to embedding a single sentence and reading its embedding dimension.
    if fallback:
        from raglite._embed import embed_sentences

        warnings.warn(
            f"Could not determine the embedding dimension of {config.embedder} from LiteLLM's model_info, using fallback.",
            stacklevel=2,
        )
        fallback_embeddings = embed_sentences(["Hello world"], config=config)
        return fallback_embeddings.shape[1]
    error_message = f"Could not determine the embedding dimension of {config.embedder}."
    raise ValueError(error_message)
