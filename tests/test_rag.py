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
