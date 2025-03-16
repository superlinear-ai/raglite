"""Test RAGLite's upgraded chatml-function-calling llama-cpp-python chat handler."""

import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from typeguard import ForwardRefPolicy, check_type

from raglite._chatml_function_calling import chatml_function_calling_with_streaming
from raglite._lazy_llama import (
    Llama,
    llama_supports_gpu_offload,
    llama_types,
)


def is_accelerator_available() -> bool:
    """Check if an accelerator is available."""
    return llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 8  # noqa: PLR2004


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(True, id="stream=True"),
        pytest.param(False, id="stream=False"),
    ],
)
@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("none", id="tool_choice=none"),
        pytest.param("auto", id="tool_choice=auto"),
        pytest.param(
            {"type": "function", "function": {"name": "get_weather"}}, id="tool_choice=fixed"
        ),
    ],
)
@pytest.mark.parametrize(
    "user_prompt_expected_tool_calls",
    [
        pytest.param(
            ("Is 7 a prime number?", 0),
            id="expected_tool_calls=0",
        ),
        pytest.param(
            ("What's the weather like in Paris today?", 1),
            id="expected_tool_calls=1",
        ),
        pytest.param(
            ("What's the weather like in Paris today? What about New York?", 2),
            id="expected_tool_calls=2",
        ),
    ],
)
@pytest.mark.parametrize(
    "llm_repo_id",
    [
        pytest.param("bartowski/Llama-3.2-3B-Instruct-GGUF", id="llama_3.2_3B"),
        pytest.param(
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            id="llama_3.1_8B",
            marks=pytest.mark.skipif(
                not is_accelerator_available(), reason="Accelerator not available"
            ),
        ),
    ],
)
def test_llama_cpp_python_tool_use(
    llm_repo_id: str,
    user_prompt_expected_tool_calls: tuple[str, int],
    tool_choice: llama_types.ChatCompletionToolChoiceOption,
    stream: bool,  # noqa: FBT001
) -> None:
    """Test the upgraded chatml-function-calling llama-cpp-python chat handler."""
    user_prompt, expected_tool_calls = user_prompt_expected_tool_calls
    if isinstance(tool_choice, dict) and expected_tool_calls == 0:
        pytest.skip("Nonsensical")
    llm = Llama.from_pretrained(
        repo_id=llm_repo_id,
        filename="*Q4_K_M.gguf",
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        chat_handler=chatml_function_calling_with_streaming,
    )
    messages: list[llama_types.ChatCompletionRequestMessage] = [
        {"role": "user", "content": user_prompt}
    ]
    tools: list[llama_types.ChatCompletionTool] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "A city name."}},
                },
            },
        }
    ]
    response = llm.create_chat_completion(
        messages=messages, tools=tools, tool_choice=tool_choice, stream=stream
    )
    if stream:
        response = cast("Iterator[llama_types.CreateChatCompletionStreamResponse]", response)
        num_tool_calls = 0
        for chunk in response:
            check_type(chunk, llama_types.CreateChatCompletionStreamResponse)
            tool_calls = chunk["choices"][0]["delta"].get("tool_calls")
            if isinstance(tool_calls, list):
                num_tool_calls = max(tool_call["index"] for tool_call in tool_calls) + 1
        assert num_tool_calls == (expected_tool_calls if tool_choice != "none" else 0)
    else:
        response = cast("llama_types.CreateChatCompletionResponse", response)
        check_type(
            response,
            llama_types.CreateChatCompletionResponse,
            forward_ref_policy=ForwardRefPolicy.IGNORE,
        )
        if expected_tool_calls == 0 or tool_choice == "none":
            assert response["choices"][0]["message"].get("tool_calls") is None
        else:
            assert len(response["choices"][0]["message"]["tool_calls"]) == expected_tool_calls
            assert all(
                tool_call["function"]["name"] == tools[0]["function"]["name"]
                for tool_call in response["choices"][0]["message"]["tool_calls"]
            )
