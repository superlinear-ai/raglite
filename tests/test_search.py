"""Test RAGLite's search functionality."""

import pytest

from raglite import (
    RAGLiteConfig,
    hybrid_search,
    keyword_search,
    retrieve_chunk_spans,
    retrieve_chunks,
    vector_search,
)
from raglite._database import Chunk, ChunkSpan, Document
from raglite._typing import BasicSearchMethod


@pytest.fixture(
    params=[
        pytest.param(keyword_search, id="keyword_search"),
        pytest.param(vector_search, id="vector_search"),
        pytest.param(hybrid_search, id="hybrid_search"),
    ],
)
def search_method(
    request: pytest.FixtureRequest,
) -> BasicSearchMethod:
    """Get a search method to test RAGLite with."""
    search_method: BasicSearchMethod = request.param
    return search_method


def test_search(raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod) -> None:
    """Test searching for a query."""
    # Search for a query.
    query = "What does it mean for two events to be simultaneous?"
    num_results = 5
    chunk_ids, scores = search_method(query, num_results=num_results, config=raglite_test_config)
    assert len(chunk_ids) == len(scores) == num_results
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)
    # Retrieve the chunks.
    chunks = retrieve_chunks(chunk_ids, config=raglite_test_config)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk_id == chunk.id for chunk_id, chunk in zip(chunk_ids, chunks, strict=True))
    assert any("Definition of Simultaneity" in str(chunk) for chunk in chunks), (
        "Expected 'Definition of Simultaneity' in chunks but got:\n"
        + "\n".join(f"- Chunk {i + 1}:\n{chunk!s}\n{'-' * 80}" for i, chunk in enumerate(chunks))
    )
    assert all(isinstance(chunk.document, Document) for chunk in chunks)
    # Extend the chunks with their neighbours and group them into contiguous segments.
    chunk_spans = retrieve_chunk_spans(chunk_ids, neighbors=(-1, 1), config=raglite_test_config)
    assert all(isinstance(chunk_span, ChunkSpan) for chunk_span in chunk_spans)
    assert all(isinstance(chunk_span.document, Document) for chunk_span in chunk_spans)
    chunk_spans = retrieve_chunk_spans(chunks, neighbors=(-1, 1), config=raglite_test_config)
    assert all(isinstance(chunk_span, ChunkSpan) for chunk_span in chunk_spans)
    assert all(isinstance(chunk_span.document, Document) for chunk_span in chunk_spans)


def test_search_no_results(
    raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Test searching for a query with no keyword search results."""
    query = "supercalifragilisticexpialidocious"
    num_results = 5
    chunk_ids, scores = search_method(query, num_results=num_results, config=raglite_test_config)
    num_results_expected = 0 if search_method == keyword_search else num_results
    assert len(chunk_ids) == len(scores) == num_results_expected
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)


def test_search_empty_database(llm: str, embedder: str, search_method: BasicSearchMethod) -> None:
    """Test searching for a query with an empty database."""
    raglite_test_config = RAGLiteConfig(db_url="duckdb:///:memory:", llm=llm, embedder=embedder)
    query = "supercalifragilisticexpialidocious"
    num_results = 5
    chunk_ids, scores = search_method(query, num_results=num_results, config=raglite_test_config)
    num_results_expected = 0
    assert len(chunk_ids) == len(scores) == num_results_expected
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)
