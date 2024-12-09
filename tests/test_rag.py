"""Test RAGLite's RAG functionality."""

import json

from raglite import (
    RAGLiteConfig,
    hybrid_search,
    rerank_chunks,
    retrieve_rag_context,
)
from raglite._database import ChunkSpan
from raglite._rag import compose_rag_messages, rag

# def test_rag(raglite_test_config: RAGLiteConfig) -> None:
#     """Test Retrieval-Augmented Generation."""
#     # Answer a question with RAG.
#     user_prompt = "What does it mean for two events to be simultaneous?"
#     chunk_spans = retrieve_rag_context(
#         query=user_prompt,
#         search=hybrid_search,
#         rerank=rerank_chunks,
#         config=raglite_test_config,
#     )
#     messages = compose_rag_messages(
#         user_prompt,
#         context=chunk_spans,
#     )
#     stream = rag(messages, config=raglite_test_config)
#     answer = ""
#     for update in stream:
#         assert isinstance(update, str)
#         answer += update
#     assert "simultaneous" in answer.lower()


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
    user_prompt = "Is 7 a prime number? Answer with Yes or No only."
    messages = [{"role": "user", "content": user_prompt}]
    chunk_spans = []
    stream = rag(messages, on_retrieval=lambda x: chunk_spans.extend(x), config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "yes" in answer.lower()
    # Verify that no RAG context was retrieved.
    if raglite_test_config.llm.startswith("llama-cpp-python"):
        # Llama.cpp does not support streaming tool_choice="auto" yet, so instead we verify that the
        # LLM indicates that the tool call request may be skipped by checking that content is empty.
        assert [msg["role"] for msg in messages] == ["user", "assistant", "tool", "assistant"]
        assert not json.loads(messages[-2]["content"])
    else:
        assert [msg["role"] for msg in messages] == ["user", "assistant"]
    assert not chunk_spans
