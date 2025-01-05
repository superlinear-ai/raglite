"""Test RAGLite's RAG functionality."""

import json

from raglite import (
    RAGLiteConfig,
    create_rag_instruction,
    retrieve_rag_context,
)
from raglite._database import ChunkSpan
from raglite._rag import rag


def test_rag_manual(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation with manual retrieval."""
    # Answer a question with manual RAG.
    user_prompt = "How does Einstein define 'simultaneous events' in his special relativity paper?"
    chunk_spans = retrieve_rag_context(query=user_prompt, config=raglite_test_config)
    messages = [create_rag_instruction(user_prompt, context=chunk_spans)]
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
    chunk_spans = []
    stream = rag(messages, on_retrieval=lambda x: chunk_spans.extend(x), config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "event" in answer.lower()
    # Verify that RAG context was retrieved automatically.
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(messages[-2]["content"])
    assert chunk_spans
    assert all(isinstance(chunk_span, ChunkSpan) for chunk_span in chunk_spans)


def test_rag_auto_without_retrieval(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation with automatic retrieval."""
    # Answer a question that does not require RAG.
    user_prompt = "Is 7 a prime number?"
    messages = [{"role": "user", "content": user_prompt}]
    chunk_spans = []
    stream = rag(messages, on_retrieval=lambda x: chunk_spans.extend(x), config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    # Verify that no RAG context was retrieved.
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert not chunk_spans
