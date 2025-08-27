"""Test RAGLite's search functionality."""

import pytest

from raglite import (
    Document,
    RAGLiteConfig,
    hybrid_search,
    insert_documents,
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


def test_search_with_metadata_filter(
    raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Test searching with metadata filtering."""
    # Insert test documents with metadata.
    test_docs = [
        Document.from_text("Python guide", filename="python.md", user_id="user_123"),
        Document.from_text("JavaScript guide", filename="js.md", user_id="user_456"),
    ]
    insert_documents(test_docs, config=raglite_test_config)
    # Test search with user_id filter.
    metadata_filter = {"user_id": "user_123"}
    filtered_results, filtered_scores = search_method(
        "guide", num_results=10, metadata_filter=metadata_filter, config=raglite_test_config
    )
    assert len(filtered_results) == len(filtered_scores)
    assert all(isinstance(chunk_id, str) for chunk_id in filtered_results)
    assert all(isinstance(score, float) for score in filtered_scores)
    # Verify chunks belong to the correct user.
    if filtered_results:
        chunks = retrieve_chunks(filtered_results, config=raglite_test_config)
        for chunk in chunks:
            if "user_id" in chunk.metadata_:
                assert chunk.metadata_["user_id"] == "user_123"


def test_search_with_multiple_metadata_filters(
    raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Test searching with multiple metadata filters."""
    # Insert test document with multiple metadata fields.
    test_doc = Document.from_text(
        "Python guide",
        filename="python.md",
        user_id="user_123",
        category="programming",
    )
    insert_documents([test_doc], config=raglite_test_config)
    # Test with multiple metadata fields.
    metadata_filter = {"user_id": "user_123", "category": "programming"}
    results, scores = search_method(
        "guide", num_results=10, metadata_filter=metadata_filter, config=raglite_test_config
    )
    assert len(results) == len(scores)
    assert all(isinstance(chunk_id, str) for chunk_id in results)
    assert all(isinstance(score, float) for score in scores)
    # Verify chunks match all filters.
    if results:
        chunks = retrieve_chunks(results, config=raglite_test_config)
        for chunk in chunks:
            if "user_id" in chunk.metadata_ and "category" in chunk.metadata_:
                assert chunk.metadata_["user_id"] == "user_123"
                assert chunk.metadata_["category"] == "programming"


def test_search_with_nonexistent_metadata_filter(
    raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Test searching with metadata filter that matches no documents."""
    metadata_filter = {"user_id": "nonexistent_user"}
    results, scores = search_method(
        "guide", num_results=10, metadata_filter=metadata_filter, config=raglite_test_config
    )
    # Should return no results or fewer results.
    assert len(results) == len(scores)
    assert all(isinstance(chunk_id, str) for chunk_id in results)
    assert all(isinstance(score, float) for score in scores)
