import json
from collections.abc import Callable, Generator
from typing import Any, TypedDict

from litellm import (  # type: ignore[attr-defined]
    ChatCompletionMessageToolCall,
    CustomStreamWrapper,
    stream_chunk_builder,
)


class ToolParameter(TypedDict, total=False):
    type: str
    description: str | None


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, ToolParameter],
        call: Callable[..., Any],
        required_params: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required_params = required_params or list(parameters.keys())
        self.call = call

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                    "additionalProperties": self.parameters.get("additionalProperties", False),
                },
            },
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.call(**kwargs)


def _run_tool(
    tool_call: ChatCompletionMessageToolCall,
    *,
    tools: list[Tool] | None = None,
) -> Any:
    """Run a tool given a tool call and a list of available tools."""
    tool_mapping = {tool.name: tool for tool in tools} if tools else {}

    if tool_call.function.name not in tool_mapping:
        error_message = f"Unknown function `{tool_call.function.name}`."
        raise ValueError(error_message)
    arguments = json.loads(tool_call.function.arguments)
    tool = tool_mapping[tool_call.function.name]
    result = tool(**arguments)
    return result


def _create_tool_response_message(
    tool_call: ChatCompletionMessageToolCall, content: Any
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": content,
    }


def process_stream(
    stream: CustomStreamWrapper,
    *,
    messages: list[dict[str, Any]],
    tools: list[Tool] | None,
    on_tool_call: Callable[[ChatCompletionMessageToolCall], Any] | None = None,
) -> Generator[str]:
    """Read the stream and append the assistant responses to the messages array in place."""
    chunks = []

    for chunk in stream:
        chunks.append(chunk)
        if isinstance(token := chunk.choices[0].delta.content, str):
            yield token

    # Check if there are tools to be called.
    response = stream_chunk_builder(chunks, messages)

    # Append the assistant response to the message array.
    messages.append(response.choices[0].message.to_dict())  # type: ignore[union-attr]
    tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]
    if tool_calls:
        for tool_call in tool_calls:
            on_tool_call(tool_call) if on_tool_call else None
            response = _run_tool(tool_call, tools=tools)
            messages.append(_create_tool_response_message(tool_call, response))
