"""Test RAGLite's RAG functionality."""

import os
from pathlib import Path

import pytest
from llama_cpp import llama_supports_gpu_offload

from raglite import RAGLiteConfig, hybrid_search, insert_document, rag, retrieve_segments


def is_accelerator_available() -> bool:
    """Check if an accelerator is available."""
    return llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 8  # noqa: PLR2004


def test_insert_index_search(raglite_test_config: RAGLiteConfig) -> None:
    """Test inserting a document, updating the indexes, and searching for a query."""
    # Insert a document and update the index.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=raglite_test_config)
    # Search for a query.
    query = "What does it mean for two events to be simultaneous?"
    chunk_ids, scores = hybrid_search(query, config=raglite_test_config)
    assert len(chunk_ids) == len(scores)
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)
    # Group the chunks into segments and retrieve them.
    segments = retrieve_segments(chunk_ids, neighbors=None, config=raglite_test_config)
    assert all(isinstance(segment, str) for segment in segments)
    assert "Definition of Simultaneity" in "".join(segments[:2])


@pytest.mark.skipif(not is_accelerator_available(), reason="No accelerator available")
def test_rag(raglite_test_config: RAGLiteConfig) -> None:
    """Test Retrieval-Augmented Generation."""
    # Insert a document and update the index.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=raglite_test_config)
    # Answer a question with RAG.
    prompt = "What does it mean for two events to be simultaneous?"
    stream = rag(prompt, search=hybrid_search, config=raglite_test_config)
    answer = ""
    for update in stream:
        assert isinstance(update, str)
        answer += update
    assert "simultaneous" in answer.lower()
