"""Tests for tools."""

from typing import Any

import pytest
from litellm import ChatCompletionMessageToolCall  # type: ignore[attr-defined]

from raglite._tools import Tool, _run_tool


def example_tool_function(**kwargs: Any) -> str:
    """Return the input arguments."""
    return f"Result: {kwargs}"


def test_tool_basic() -> None:
    """Test the Tool class."""
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"param1": {"type": "string", "description": "First parameter"}},
        call=example_tool_function,
    )

    # Test initialization
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"

    # Test JSON output
    json_output = tool.to_json()
    assert json_output["name"] == "test_tool"
    assert json_output["type"] == "function"
    assert json_output["function"]["parameters"]["required"] == ["param1"]

    # Test calling
    result = tool(param1="test_value")
    assert result == "Result: {'param1': 'test_value'}"


def test_run_tool() -> None:
    """Test the _run_tool function."""
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"param1": {"type": "string"}},
        call=example_tool_function,
    )

    # Create mock tool call
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        type="function",
        function={"name": "test_tool", "arguments": '{"param1": "test_value"}'},
    )

    # Test successful execution
    result = _run_tool(tool_call, tools=[tool])
    assert result == "Result: {'param1': 'test_value'}"

    # Test unknown tool
    bad_tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        type="function",
        function={"name": "unknown_tool", "arguments": '{"param1": "test_value"}'},
    )

    with pytest.raises(ValueError, match="Unknown function"):
        _run_tool(bad_tool_call, tools=[tool])
