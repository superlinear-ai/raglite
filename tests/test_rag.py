"""Test RAGLite's RAG functionality."""

import json
from types import SimpleNamespace

from raglite import (
    RAGLiteConfig,
    add_context,
    retrieve_context,
)
from raglite._database import ChunkSpan
from raglite._rag import _extract_tool_query, _run_tools, rag


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
    user_prompt = "How does Einstein define 'simultaneous events' in his special relativity paper?"
    messages = [{"role": "user", "content": user_prompt}]
    chunk_spans: list[ChunkSpan] = []
    stream = rag(messages, on_retrieval=chunk_spans.extend, config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "event" in answer.lower()
    # Verify that RAG context was retrieved automatically.
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(messages[-2]["content"])
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


def test_extract_tool_query() -> None:
    """Extract tool query from serialized function arguments."""
    assert _extract_tool_query('{"query":"When was Einstein born?"}') == "When was Einstein born?"
    assert _extract_tool_query('{"query": 123}') is None
    assert _extract_tool_query("{") is None


def test_run_tools_returns_message_per_tool_call(monkeypatch) -> None:
    """Return one tool message per tool call."""
    def fake_run_tool(tool_call, config):
        return tool_call.id, []

    monkeypatch.setattr("raglite._rag._run_tool", fake_run_tool)
    config = RAGLiteConfig(llm="gpt-4o-mini", embedder="text-embedding-3-small", reranker=None)

    tool_calls = [
        SimpleNamespace(
            id="search_call",
            function=SimpleNamespace(
                name="search_knowledge_base", arguments='{"query":"Where was Einstein born?"}'
            ),
        ),
        SimpleNamespace(
            id="direct_query_call",
            function=SimpleNamespace(name="query_knowledge_base", arguments='{"query":"Ulm"}'),
        ),
    ]
    tool_messages = _run_tools(
        tool_calls,  # type: ignore[arg-type]
        on_retrieval=None,
        config=config,
        messages=[],
        max_workers=1,
    )

    expected_tool_messages = 2
    assert len(tool_messages) == expected_tool_messages
