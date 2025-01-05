"""Upgrade of llama-cpp-python's chatml-function-calling chat handler.

Changes:
1. General:
    a. ✨ If no system message is supplied, add an empty system message to hold the tool metadata.
    b. ✨ Add function descriptions to the system message so that tool use is better informed (fixes https://github.com/abetlen/llama-cpp-python/issues/1869).
    c. ✨ Replace `print` statements relating to JSON grammars with `RuntimeWarning` warnings.
    d. ✅ Add tests with fairly broad coverage of the different scenarios.
4. Case "Tool choice by user":
    a. ✨ Add support for more than one function call by making this a special case of "Automatic tool choice" with a single tool (subsumes https://github.com/abetlen/llama-cpp-python/pull/1503).
5. Case "Automatic tool choice -> respond with a message":
    a. ✨ Use user-defined `stop` and `max_tokens`.
    b. 🐛 Replace incorrect use of follow-up grammar with user-defined grammar.
6. Case "Automatic tool choice -> one or more function calls":
    a. ✨ Add support for streaming the function calls (fixes https://github.com/abetlen/llama-cpp-python/issues/1883).
    b. ✨ Make tool calling more robust by giving the LLM an explicit way to terminate the tool calls by wrapping them in a `<function_calls></function_calls>` block.
    c. 🐛 Add missing ":" stop token to determine whether to continue with another tool call, which prevented parallel function calling (fixes https://github.com/abetlen/llama-cpp-python/issues/1756).
    d. ✨ Set temperature=0 to determine whether to continue with another tool call, similar to the initial decision on whether to call a tool.
"""
# This file uses old-style type hints and ignores certain ruff rules to minimise changes w.r.t. the original implementation:
# ruff: noqa: C901, PLR0913, PLR0912, PLR0915, UP006, UP007, FBT001, FBT002, B006, TRY003, EM102, BLE001, PT018, W505

import json
import warnings
from typing import (  # noqa: UP035
    Any,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import llama, llama_grammar, llama_types
from llama_cpp.llama_chat_format import (
    _convert_completion_to_chat,
    _convert_completion_to_chat_function,
    _grammar_for_response_format,
)


def _accumulate_chunks(
    chunks_iterator: Iterator[llama_types.CreateCompletionStreamResponse],
    chunks_list: List[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.CreateCompletionStreamResponse]:
    for chunk in chunks_iterator:
        chunks_list.append(chunk)
        yield chunk


def _convert_chunks_to_completion(
    chunks: List[llama_types.CreateCompletionStreamResponse],
) -> llama_types.CreateCompletionResponse:
    """Convert a list of completion chunks to a completion."""
    # Accumulate completion response values
    text: str = ""
    finish_reason: Optional[str] = None
    logprobs: Optional[llama_types.CompletionLogprobs] = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    completion_id: Optional[str] = None
    completion_model: Optional[str] = None
    completion_created: Optional[int] = None
    for chunk in chunks:
        # Extract the id, model, and created values from the first chunk
        if completion_id is None:
            completion_id = chunk["id"]
            completion_model = chunk["model"]
            completion_created = chunk["created"]
        # Extract the usage if present in the chunk
        usage = chunk.get("usage")
        if usage:
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
        # Accumulate the chunk text
        choice = chunk["choices"][0]
        text += choice.get("text", "")
        # Extract the finish_reason and logprobs if present in the chunk
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if choice.get("logprobs"):
            logprobs = choice["logprobs"]
    # Create the completion response
    completion: llama_types.CreateCompletionResponse = {
        "id": completion_id or "unknown_id",
        "object": "text_completion",
        "created": completion_created or 0,
        "model": completion_model or "unknown_model",
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": logprobs,  # TODO: Improve accumulation of logprobs
                "finish_reason": finish_reason,  # type: ignore[typeddict-item]
            }
        ],
    }
    # Add usage section if present in the chunks
    if (prompt_tokens + completion_tokens + total_tokens) > 0:
        completion["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return completion


def _stream_tool_calls(
    llama: llama.Llama,
    prompt: str,
    tools: List[llama_types.ChatCompletionTool],
    tool_name: str,
    completion_kwargs: dict[str, Any],
    follow_up_gbnf_tool_grammar: str,
) -> Iterator[llama_types.CreateChatCompletionStreamResponse]:
    # Generate a tool call completions
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    completions: List[llama_types.CreateCompletionResponse] = []
    completions_tool_name: List[str] = []
    finish_reason_chat_chunk = None
    while tool is not None and len(completions) <= 16:  # noqa: PLR2004
        # Generate the parameter values for the selected tool
        prompt += f"functions.{tool_name}:\n"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            warnings.warn(
                f"Failed to parse function body as JSON schema, falling back to default grammar\n\n{e}",
                category=RuntimeWarning,
                stacklevel=2,
            )
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            **{
                **completion_kwargs,
                "max_tokens": None,
                "grammar": grammar,
            },
        )
        chunks: List[llama_types.CreateCompletionResponse] = []
        chat_chunks = _convert_completion_to_chat_function(
            tool_name,
            _accumulate_chunks(completion_or_chunks, chunks),  # type: ignore[arg-type]
            stream=True,
        )
        for chat_chunk in chat_chunks:
            # Don't return the finish_reason chunk
            if chat_chunk["choices"] and chat_chunk["choices"][0].get("finish_reason"):
                finish_reason_chat_chunk = chat_chunk
                break
            # Update this tool call's index
            if chat_chunk["choices"] and chat_chunk["choices"][0]["delta"].get("tool_calls"):
                chat_chunk["choices"][0]["delta"]["tool_calls"][0]["index"] = len(completions)
            yield chat_chunk
        completion = _convert_chunks_to_completion(chunks)
        completions.append(completion)
        completions_tool_name.append(tool_name)
        prompt += completion["choices"][0]["text"]
        prompt += "\n"
        # Determine whether to call another tool or stop
        response = cast(
            llama_types.CreateCompletionResponse,
            llama.create_completion(
                prompt=prompt,
                **{
                    **completion_kwargs,
                    "temperature": 0,
                    "stream": False,
                    "stop": [*completion_kwargs["stop"], ":", "</function_calls>"],
                    "max_tokens": None,
                    "grammar": llama_grammar.LlamaGrammar.from_string(
                        follow_up_gbnf_tool_grammar, verbose=llama.verbose
                    ),
                },
            ),
        )
        tool_name = response["choices"][0]["text"][len("functions.") :]
        tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    # Yield the finish_reason chunk
    if finish_reason_chat_chunk is not None:
        yield finish_reason_chat_chunk


def chatml_function_calling_with_streaming(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,  # type: ignore[name-defined]
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs: Any,
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:
    function_calling_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n"
        # System message
        "{% if message.role == 'system' %}"
        "{{ message.content }}"
        "{% if tool_calls %}"
        "\n\nYou have access to the following functions:\n"
        "{% for tool in tools %}"
        '\n{% if tool.function.get("description") %}/* {{ tool.function.description | trim }} */{% endif %}'
        "\nfunctions.{{ tool.function.name }}:\n"
        "{{ tool.function.parameters | tojson }}"
        "\n{% endfor %}"
        "\nYou must respond to user messages with either a single message or with one or more function calls."
        "\n\nTo respond with a message use the following format:"
        "\n\nmessage:"
        "\n<message>"
        "\n\nTo respond with one or more function calls use the following format:"
        "\n\n<function_calls>"
        "\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "\n</function_calls>"
        "{% endif %}"
        "<|im_end|>\n"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        ## Regular message
        "{% if message.content and message.content | length > 0 %}"
        "{% if tool_calls %}"
        "message:\n"
        "{% endif %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        ## Function calls
        "{% if 'tool_calls' in message %}"
        "{% for tool_call in message.tool_calls %}"
        "functions.{{ tool_call.function.name }}:\n"
        "{{ tool_call.function.arguments }}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    template_renderer = ImmutableSandboxedEnvironment(
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        undefined=jinja2.StrictUndefined,
    ).from_string(function_calling_template)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [{"type": "function", "function": function} for function in functions]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (function_call in ("none", "auto")):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {"type": "function", "function": {"name": function_call["name"]}}

    # Collect the llama.create_completion keyword arguments so we don't have to repeat these with
    # each completion call
    stop = (
        [stop, "<|im_end|>", "|im_end|>"]
        if isinstance(stop, str)
        else [*stop, "<|im_end|>", "|im_end|>"]
        if stop
        else ["<|im_end|>", "|im_end|>"]
    )
    grammar = (  # It is assumed the grammar applies to messages only, not tool calls
        grammar
        if grammar is not None
        else (
            _grammar_for_response_format(response_format)
            if response_format is not None and response_format["type"] == "json_object"
            else None
        )
    )
    completion_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "typical_p": typical_p,
        "stream": stream,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repeat_penalty": repeat_penalty,
        "tfs_z": tfs_z,
        "mirostat_mode": mirostat_mode,
        "mirostat_tau": mirostat_tau,
        "mirostat_eta": mirostat_eta,
        "model": model,
        "logits_processor": logits_processor,
        "grammar": grammar,
    }

    # Case 1: No tool use
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages, tools=[], tool_calls=None, add_generation_prompt=True
        )
        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                **completion_kwargs,  # type: ignore[arg-type]
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Ensure there is a system prompt to attach the tool metadata to
    if not any(message["role"] == "system" for message in messages):
        messages = [*messages, {"role": "system", "content": ""}]

    # Case 2: Automatic or fixed tool choice
    # Case 2 step 1: Determine whether to respond with a message or a tool call
    assert (isinstance(tool_choice, str) and tool_choice == "auto") or isinstance(tool_choice, dict)
    if isinstance(tool_choice, dict):
        tools = [t for t in tools if t["function"]["name"] == tool_choice["function"]["name"]]
        assert tools
    function_names = " | ".join([f'''"functions.{t['function']['name']}:"''' for t in tools])
    prompt = template_renderer.render(
        messages=messages, tools=tools, tool_calls=True, add_generation_prompt=True
    )
    initial_gbnf_tool_grammar = (
        (
            'root ::= "<function_calls>" "\\n" functions | "message:"\n'
            f"functions ::= {function_names}\n"
        )
        if tool_choice == "auto"
        else f'root ::= "<function_calls>" "\\n" functions\nfunctions ::= {function_names}\n'
    )
    completion = cast(
        llama_types.CreateCompletionResponse,
        llama.create_completion(
            prompt=prompt,
            **{  # type: ignore[arg-type]
                **completion_kwargs,
                "temperature": 0,
                "stream": False,
                "stop": [":"],
                "max_tokens": None,
                "grammar": llama_grammar.LlamaGrammar.from_string(
                    initial_gbnf_tool_grammar, verbose=llama.verbose
                ),
            },
        ),
    )
    text = completion["choices"][0]["text"]
    tool_name = None if text.startswith("message") else text.split("\n")[-1][len("functions.") :]

    # Case 2 step 2A: Respond with a message
    if tool_name is None:
        prompt = template_renderer.render(
            messages=messages, tools=[], tool_calls=None, add_generation_prompt=True
        )
        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                **completion_kwargs,  # type: ignore[arg-type]
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2 step 2B: One or more function calls
    follow_up_gbnf_tool_grammar = (
        'root ::= functions | "</function_calls>" | "<|im_end|>"\n'
        f"functions ::= {function_names}\n"
    )
    prompt += "<function_calls>\n"
    if stream:
        return _stream_tool_calls(
            llama, prompt, tools, tool_name, completion_kwargs, follow_up_gbnf_tool_grammar
        )
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    completions: List[llama_types.CreateCompletionResponse] = []
    completions_tool_name: List[str] = []
    while tool is not None and len(completions) <= 16:  # noqa: PLR2004
        # Generate the parameter values for the selected tool
        prompt += f"functions.{tool_name}:\n"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            warnings.warn(
                f"Failed to parse function body as JSON schema, falling back to default grammar\n\n{e}",
                category=RuntimeWarning,
                stacklevel=2,
            )
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            **{  # type: ignore[arg-type]
                **completion_kwargs,
                "max_tokens": None,
                "grammar": grammar,
            },
        )
        completion = cast(llama_types.CreateCompletionResponse, completion_or_chunks)
        completions.append(completion)
        completions_tool_name.append(tool_name)
        prompt += completion["choices"][0]["text"]
        prompt += "\n"
        # Determine whether to call another tool or stop
        response = cast(
            llama_types.CreateCompletionResponse,
            llama.create_completion(
                prompt=prompt,
                **{  # type: ignore[arg-type]
                    **completion_kwargs,
                    "temperature": 0,
                    "stream": False,
                    "stop": [*completion_kwargs["stop"], ":", "</function_calls>"],  # type: ignore[misc]
                    "max_tokens": None,
                    "grammar": llama_grammar.LlamaGrammar.from_string(
                        follow_up_gbnf_tool_grammar, verbose=llama.verbose
                    ),
                },
            ),
        )
        tool_name = response["choices"][0]["text"][len("functions.") :]
        tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    # Merge the completions into a single chat completion
    chat_completion: llama_types.CreateChatCompletionResponse = {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "logprobs": completion["choices"][0]["logprobs"],
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_" + f"_{i}_" + tool_name + "_" + completion["id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": completion["choices"][0]["text"],
                            },
                        }
                        for i, (tool_name, completion) in enumerate(
                            zip(completions_tool_name, completions, strict=True)
                        )
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": sum(
                (completion["usage"]["completion_tokens"] if "usage" in completion else 0)
                for completion in completions
            ),
            "prompt_tokens": sum(
                completion["usage"]["prompt_tokens"] if "usage" in completion else 0
                for completion in completions
            ),
            "total_tokens": sum(
                completion["usage"]["total_tokens"] if "usage" in completion else 0
                for completion in completions
            ),
        },
    }
    if len(completions) == 1:
        single_function_call: llama_types.ChatCompletionResponseFunctionCall = {
            "name": tool_name,
            "arguments": completions[0]["choices"][0]["text"],
        }
        chat_completion["choices"][0]["message"]["function_call"] = single_function_call
    return chat_completion
