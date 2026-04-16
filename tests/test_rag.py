"""Test RAGLite's RAG functionality."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from raglite import (
    RAGLiteConfig,
    add_context,
    retrieve_context,
)
from raglite._database import ChunkSpan
from raglite._rag import _run_tool, rag
from raglite._typing import MetadataFilter  # noqa: TC001


def test_rag_manual(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation with manual retrieval."""
    # Answer a question with manual RAG.
    user_prompt = "How does Einstein define 'simultaneous events' in his special relativity paper?"
    chunk_spans = retrieve_context(query=user_prompt, config=raglite_test_config)
    messages = [add_context(user_prompt, context=chunk_spans, config=raglite_test_config)]
    stream = rag(messages, config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "event" in answer.lower()
    # Verify that no RAG context was retrieved through tool use.
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_rag_auto_with_retrieval(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation with automatic retrieval."""
    # Answer a question that requires RAG.
    user_prompt = "How does Einstein define 'simultaneous events' in his special relativity paper? do not guess and provide me with proof via retrieval"
    messages = [{"role": "user", "content": user_prompt}]
    chunk_spans: list[ChunkSpan] = []
    stream = rag(messages, on_retrieval=chunk_spans.extend, config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "event" in answer.lower()
    # Verify that RAG context was retrieved automatically.
    roles = [message["role"] for message in messages]
    assert roles[0] == "user"
    assert roles[-1] == "assistant"
    assert "tool" in roles  # At least one retrieval happened.
    if not raglite_test_config.llm.startswith("llama-cpp-python"):
        assert chunk_spans
    assert all(isinstance(chunk_span, ChunkSpan) for chunk_span in chunk_spans)


def test_rag_auto_without_retrieval(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation with automatic retrieval."""
    # Answer a question that does not require RAG.
    user_prompt = "Is 7 a prime number?"
    messages = [{"role": "user", "content": user_prompt}]
    chunk_spans: list[ChunkSpan] = []
    stream = rag(messages, on_retrieval=chunk_spans.extend, config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    # Verify that no RAG context was retrieved.
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert not chunk_spans


def test_retrieve_context_self_query(raglite_test_config: RAGLiteConfig) -> None:
    """Test retrieve_context with self_query functionality."""
    from dataclasses import replace

    new_config = replace(raglite_test_config, self_query=True)
    query = "What does Albert Einstein's paper say about time dilation?"
    chunk_spans = retrieve_context(query=query, num_chunks=5, config=new_config)
    assert all(isinstance(chunk_span, ChunkSpan) for chunk_span in chunk_spans)
    for chunk_span in chunk_spans:
        assert chunk_span.document.metadata_.get("type") == ["Paper"], (
            f"Expected type='Paper', got {chunk_span.document.metadata_.get('type')}"
        )
        assert chunk_span.document.metadata_.get("author") == ["Albert Einstein"], (
            f"Expected author='Albert Einstein', got {chunk_span.document.metadata_.get('author')}"
        )


def test_agentic_search_threads_metadata_filter_to_nested_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass metadata filters from search_knowledge_base down to nested tool calls."""
    config = RAGLiteConfig(
        llm="gpt-5-mini",
        embedder="text-embedding-3-small",
        db_url="duckdb:///:memory:",
    )
    metadata_filter: MetadataFilter = {"topic": ["Physics"]}
    nested_tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="query_knowledge_base",
            arguments=json.dumps({"query": "What is time dilation?"}),
        ),
        id="query_call_id",
    )
    search_tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="search_knowledge_base",
            arguments=json.dumps({"query": "Explain Einstein's time dilation."}),
        ),
        id="search_call_id",
    )

    def _make_response(
        tool_calls: list[Any] | None,
    ) -> Any:
        message = SimpleNamespace(
            tool_calls=tool_calls,
            to_dict=lambda: {"role": "assistant", "content": ""},
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completion_responses = [
        _make_response([nested_tool_call]),
        _make_response(None),
    ]

    def fake_completion(**_: Any) -> Any:
        return completion_responses.pop(0)

    observed_metadata_filters: list[MetadataFilter | None] = []

    def fake_run_tools(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        tool_calls = args[0]
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "query_knowledge_base"
        observed_metadata_filters.append(kwargs.get("metadata_filter"))
        return []

    monkeypatch.setattr("raglite._rag.completion", fake_completion)
    monkeypatch.setattr("raglite._rag._run_tools", fake_run_tools)

    tool_id, chunk_spans = _run_tool(
        search_tool_call,
        config,
        metadata_filter=metadata_filter,
    )

    assert tool_id == "search_call_id"
    assert chunk_spans == []
    assert observed_metadata_filters == [metadata_filter]


def test_query_tool_call_passes_metadata_filter_to_retrieve_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass metadata_filter to retrieve_context when running query_knowledge_base."""
    config = RAGLiteConfig(
        llm="gpt-5-mini",
        embedder="text-embedding-3-small",
        db_url="duckdb:///:memory:",
    )
    metadata_filter: MetadataFilter = {"type": ["Paper"], "author": ["Albert Einstein"]}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="query_knowledge_base",
            arguments=json.dumps({"query": "How is simultaneity defined?"}),
        ),
        id="query_call_id",
    )
    observed_kwargs: dict[str, Any] = {}

    def fake_retrieve_context(**kwargs: Any) -> list[ChunkSpan]:
        observed_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("raglite._rag.retrieve_context", fake_retrieve_context)

    tool_id, chunk_spans = _run_tool(
        tool_call,
        config,
        metadata_filter=metadata_filter,
    )

    assert tool_id == "query_call_id"
    assert chunk_spans == []
    assert observed_kwargs["metadata_filter"] == metadata_filter


def test_sub_agent_deduplicates_chunk_spans_by_chunk_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop fully redundant spans and keep partially novel spans in sub-agent search."""
    config = RAGLiteConfig(
        llm="gpt-5-mini",
        embedder="text-embedding-3-small",
        db_url="duckdb:///:memory:",
    )
    search_tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="search_knowledge_base",
            arguments=json.dumps({"query": "Explain relativity."}),
        ),
        id="search_call_id",
    )

    def _make_response(tool_calls: list[Any] | None) -> Any:
        message = SimpleNamespace(
            tool_calls=tool_calls,
            to_dict=lambda: {"role": "assistant", "content": ""},
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    nested_tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="query_knowledge_base",
            arguments=json.dumps({"query": "Q"}),
        ),
        id="query_call_id",
    )
    completion_responses = [
        _make_response([nested_tool_call]),
        _make_response([nested_tool_call]),
        _make_response(None),
    ]

    def fake_completion(**_: Any) -> Any:
        return completion_responses.pop(0)

    def make_chunk_span(*chunk_ids: str) -> Any:
        return SimpleNamespace(chunks=[SimpleNamespace(id=chunk_id) for chunk_id in chunk_ids])

    first_iteration_spans = [
        make_chunk_span("A", "B"),
    ]
    second_iteration_spans = [
        make_chunk_span("A", "B"),
        make_chunk_span("B", "C"),
    ]
    tool_results_by_iteration = [first_iteration_spans, second_iteration_spans]

    def fake_run_tools(*args: Any, **_: Any) -> list[dict[str, Any]]:
        on_retrieval = args[1]
        on_retrieval(tool_results_by_iteration.pop(0))
        return []

    monkeypatch.setattr("raglite._rag.completion", fake_completion)
    monkeypatch.setattr("raglite._rag._run_tools", fake_run_tools)

    _, chunk_spans = _run_tool(search_tool_call, config)

    actual_chunk_id_sequences = [
        [chunk.id for chunk in chunk_span.chunks] for chunk_span in chunk_spans
    ]
    assert actual_chunk_id_sequences == [["A", "B"], ["B", "C"]]


def test_rag_does_not_mutate_caller_messages_on_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep caller messages unchanged when an exception happens mid-rag."""
    config = RAGLiteConfig(
        llm="gpt-5-mini",
        embedder="text-embedding-3-small",
        db_url="duckdb:///:memory:",
    )
    messages = [{"role": "user", "content": "Hello"}]
    original_messages = list(messages)

    def fake_stream(*_: Any, **__: Any) -> Any:
        error_message = "stream failure"
        raise RuntimeError(error_message)

    monkeypatch.setattr("raglite._rag._stream_rag_response", fake_stream)

    with pytest.raises(RuntimeError, match="stream failure"):
        list(rag(messages, config=config))
    assert messages == original_messages
