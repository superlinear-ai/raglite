"""Test RAGLite's RAG functionality."""

import os

import pytest
from llama_cpp import llama_supports_gpu_offload

from raglite import (
    RAGLiteConfig,
    hybrid_search,
    rerank_chunks,
    retrieve_rag_context,
)
from raglite._rag import compose_rag_messages, rag


def is_accelerator_available() -> bool:
    """Check if an accelerator is available."""
    return llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 8  # noqa: PLR2004


@pytest.mark.skipif(not is_accelerator_available(), reason="No accelerator available")
def test_rag(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation."""
    # Answer a question with RAG.
    user_prompt = "What does it mean for two events to be simultaneous?"
    chunk_spans = retrieve_rag_context(
        query=user_prompt,
        search=hybrid_search,
        rerank=rerank_chunks,
        config=raglite_test_config,
    )
    messages = compose_rag_messages(
        user_prompt,
        context=chunk_spans,
    )
    stream = rag(messages, config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "simultaneous" in answer.lower()
