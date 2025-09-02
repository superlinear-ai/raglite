"""Test RAGLite's search functionality."""

import pytest

from raglite import (
    Document,
    RAGLiteConfig,
    hybrid_search,
    keyword_search,
    retrieve_chunk_spans,
    retrieve_chunks,
    vector_search,
)
from raglite._database import Chunk, ChunkSpan
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


def test_search_metadata_filter(
    raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Test searching with metadata filtering that should return results."""
    query = "What does it mean for two events to be simultaneous?"
    num_results = 5
    # Filter for Physics paper (should match Einstein's special relativity paper)
    metadata_filter = {"type": "Paper", "topic": "Physics"}
    chunk_ids, scores = search_method(
        query, num_results=num_results, metadata_filter=metadata_filter, config=raglite_test_config
    )
    # Assert basic properties of the results
    assert len(chunk_ids) == len(scores)
    assert len(chunk_ids) > 0, "Expected results when filtering for Physics papers"
    assert len(chunk_ids) <= num_results, "Should not exceed requested number of results"
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)
    # Retrieve and verify the chunks match the metadata filter
    chunks = retrieve_chunks(chunk_ids, config=raglite_test_config)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    for chunk in chunks:
        assert chunk.metadata_.get("type") == "Paper", (
            f"Expected type='Paper', got {chunk.metadata_.get('type')}"
        )
        assert chunk.metadata_.get("topic") == "Physics", (
            f"Expected topic='Physics', got {chunk.metadata_.get('topic')}"
        )

    # Test with different topic that should return no results
    metadata_filter_no_match = {"type": "Paper", "topic": "Mathematics"}
    chunk_ids_no_match, scores_no_match = search_method(
        query,
        num_results=num_results,
        metadata_filter=metadata_filter_no_match,
        config=raglite_test_config,
    )
    assert len(chunk_ids_no_match) == len(scores_no_match) == 0, (
        "Expected no results when filtering for Mathematics papers"
    )
