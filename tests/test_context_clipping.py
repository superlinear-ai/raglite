"""Keep tool-call exchanges valid when trimming conversation history."""

import asyncio
from typing import Any

import pytest

from raglite import RAGLiteConfig
from raglite._rag import _async_stream_rag_response, _clip, _stream_rag_response


@pytest.fixture
def tool_history() -> list[dict[str, Any]]:
    """Create a completed parallel retrieval followed by a new user turn."""
    return [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "search", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "first result"},
        {"role": "tool", "tool_call_id": "b", "content": "second result"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]


@pytest.mark.parametrize("budget", [50, 71])
def test_clipping_drops_orphaned_tool_results(
    tool_history: list[dict[str, Any]], budget: int
) -> None:
    """A suffix starting inside tool results must not be sent without their call."""
    original_history = tool_history.copy()
    clipped = _clip(tool_history, budget)
    assert clipped == tool_history[4:]
    assert tool_history == original_history


def test_clipping_keeps_complete_tool_exchange(
    tool_history: list[dict[str, Any]],
) -> None:
    """Retain a whole call and its results when the budget permits it."""
    assert _clip(tool_history, 145) == tool_history[1:]


def test_clipping_without_user_keeps_last_complete_exchange(
    tool_history: list[dict[str, Any]],
) -> None:
    """The best-effort fallback must preserve the call as well as its results."""
    exchange = tool_history[1:4]
    assert _clip(exchange, 10) == exchange


def test_clipping_preserves_user_fallback(tool_history: list[dict[str, Any]]) -> None:
    """Keep the existing latest-user fallback when even that message is too large."""
    assert _clip(tool_history, 10) == [tool_history[-1]]


def test_clipping_preserves_history_that_fits(tool_history: list[dict[str, Any]]) -> None:
    """Do not discard any exchange when the full history fits."""
    assert _clip(tool_history, 200) == tool_history


@pytest.mark.parametrize("asynchronous", [False, True])
def test_streaming_sends_complete_tool_exchanges(
    tool_history: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, *, asynchronous: bool
) -> None:
    """Both streaming paths trim invalid leading tool results before provider dispatch."""
    monkeypatch.setattr("raglite._rag.get_context_size", lambda _: 96)
    captured_messages: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        captured_messages.extend(kwargs["messages"])
        return iter([])

    async def acompletion(**kwargs: Any) -> Any:
        captured_messages.extend(kwargs["messages"])

        async def stream() -> Any:
            chunks: list[str] = []
            for chunk in chunks:
                yield chunk

        return stream()

    monkeypatch.setattr("raglite._rag.completion", completion)
    monkeypatch.setattr("raglite._rag.acompletion", acompletion)
    config = RAGLiteConfig(llm="test", embedder="test", reranker=None)
    if asynchronous:

        async def consume() -> None:
            async for _ in _async_stream_rag_response(tool_history, config, [], use_tools=False):
                pass

        asyncio.run(consume())
    else:
        list(_stream_rag_response(tool_history, config, use_tools=False))
    assert captured_messages == tool_history[4:]
