"""Test RAGLite's RAG functionality."""

import os
from typing import TYPE_CHECKING

import pytest
from llama_cpp import llama_supports_gpu_offload

from raglite import RAGLiteConfig, hybrid_search, retrieve_chunks
from raglite._rag import generate, get_context_segments

if TYPE_CHECKING:
    from raglite._database import Chunk
    from raglite._typing import SearchMethod


def is_accelerator_available() -> bool:
    """Check if an accelerator is available."""
    return llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 8  # noqa: PLR2004


@pytest.mark.skipif(not is_accelerator_available(), reason="No accelerator available")
def test_rag(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation."""
    # Assemble different types of search inputs for RAG.
    prompt = "What does it mean for two events to be simultaneous?"
    search_inputs: list[SearchMethod | list[str] | list[Chunk]] = [
        hybrid_search,  # A search method as input.
        hybrid_search(prompt, config=raglite_test_config)[0],  # Chunk ids as input.
        retrieve_chunks(  # Chunks as input.
            hybrid_search(prompt, config=raglite_test_config)[0], config=raglite_test_config
        ),
    ]
    # Answer a question with RAG.
    for search_input in search_inputs:
        segments = get_context_segments(prompt, search=search_input, config=raglite_test_config)
        stream = generate(prompt, context_segments=segments, config=raglite_test_config)
        answer = ""
        for update in stream:
            assert isinstance(update, str)
            answer += update
        assert "simultaneous" in answer.lower()
