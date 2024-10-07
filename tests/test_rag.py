"""Test RAGLite's RAG functionality."""

import os

import pytest
from llama_cpp import llama_supports_gpu_offload

from raglite import RAGLiteConfig, hybrid_search, rag, retrieve_chunks


def is_accelerator_available() -> bool:
    """Check if an accelerator is available."""
    return llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 8  # noqa: PLR2004


@pytest.mark.skipif(not is_accelerator_available(), reason="No accelerator available")
def test_rag(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation."""
    # Assemble different types of search inputs for RAG.
    prompt = "What does it mean for two events to be simultaneous?"
    search_inputs = [
        hybrid_search,  # A search method as input.
        hybrid_search(prompt, config=raglite_test_config)[0],  # Chunk ids as input.
        retrieve_chunks(  # Chunks as input.
            hybrid_search(prompt, config=raglite_test_config)[0], config=raglite_test_config
        ),
    ]
    # Answer a question with RAG.
    for search_input in search_inputs:
        stream = rag(prompt, search=search_input, config=raglite_test_config)
        answer = ""
        for update in stream:
            assert isinstance(update, str)
            answer += update
        assert "simultaneous" in answer.lower()
