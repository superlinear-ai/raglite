"""Test RAGLite's search functionality."""

from typing import Any
from uuid import uuid4

import pytest

from raglite import (
    Document,
    RAGLiteConfig,
    delete_documents,
    hybrid_search,
    insert_documents,
    keyword_search,
    retrieve_chunk_spans,
    retrieve_chunks,
    vector_search,
)
from raglite._database import Chunk, ChunkSpan
from raglite._search import _self_query
from raglite._typing import BasicSearchMethod, MetadataFilter


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
    metadata_filter: MetadataFilter = {"type": "Paper", "topic": "Physics"}

    # Verify basic properties
    chunk_ids, scores = search_method(
        query, num_results=num_results, metadata_filter=metadata_filter, config=raglite_test_config
    )
    assert len(chunk_ids) == len(scores)
    assert len(chunk_ids) > 0, "Expected results when filtering for Physics papers"
    assert len(chunk_ids) <= num_results, "Should not exceed requested number of results"
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)

    # Verify chunks match metadata filter
    chunks = retrieve_chunks(chunk_ids, config=raglite_test_config)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    for chunk in chunks:
        assert chunk.metadata_.get("type") == ["Paper"], (
            f"Expected type='Paper', got {chunk.metadata_.get('type')}"
        )
        assert chunk.metadata_.get("topic") == ["Physics"], (
            f"Expected topic='Physics', got {chunk.metadata_.get('topic')}"
        )

    # Test filtering for a different topic that should return no results
    metadata_filter_empty: MetadataFilter = {"type": "Paper", "topic": "Mathematics"}
    chunk_ids_empty, scores_empty = search_method(
        query,
        num_results=num_results,
        metadata_filter=metadata_filter_empty,
        config=raglite_test_config,
    )
    assert len(chunk_ids_empty) == len(scores_empty) == 0, (
        "Expected no results when filtering for Mathematics papers"
    )


def test_search_metadata_filter_multiple_values_match_any(
    isolated_raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Match any value when a metadata field filter contains multiple values."""
    topic = f"or-filter-topic-{uuid4().hex}"
    document_ids = [f"{topic}-open", f"{topic}-music", f"{topic}-sports"]
    documents = [
        Document.from_text(
            "A short note on piano and orchestra for open-domain retrieval.",
            id=document_ids[0],
            domain="open",
            topic=topic,
        ),
        Document.from_text(
            "A short note on piano and orchestra for music-domain retrieval.",
            id=document_ids[1],
            domain="music",
            topic=topic,
        ),
        Document.from_text(
            "A short note on football and basketball for sports-domain retrieval.",
            id=document_ids[2],
            domain="sports",
            topic=topic,
        ),
    ]
    insert_documents(documents, config=isolated_raglite_test_config)

    try:
        query = "piano orchestra retrieval"
        metadata_filter: MetadataFilter = {"topic": topic, "domain": ["open", "music"]}
        chunk_ids, _ = search_method(
            query,
            num_results=5,
            metadata_filter=metadata_filter,
            config=isolated_raglite_test_config,
        )
        assert chunk_ids, (
            "Expected OR metadata filter to match documents with domain='open' and 'music'."
        )

        chunks = retrieve_chunks(chunk_ids, config=isolated_raglite_test_config)
        for chunk in chunks:
            assert chunk.metadata_.get("topic") == [topic]
            assert any(
                domain in {"open", "music"} for domain in chunk.metadata_.get("domain", [])
            ), f"Expected OR match on domain values, got {chunk.metadata_.get('domain')}"
    finally:
        delete_documents(document_ids, config=isolated_raglite_test_config)


def test_search_metadata_filter_matches_documents_with_list_metadata_values(
    isolated_raglite_test_config: RAGLiteConfig, search_method: BasicSearchMethod
) -> None:
    """Match documents when metadata values are stored as lists with multiple elements."""
    topic = f"list-domain-topic-{uuid4().hex}"
    document_ids = [f"{topic}-open-news", f"{topic}-music-arts", f"{topic}-sports-health"]
    documents = [
        Document.from_text(
            "Open and news coverage about orchestras and pianos in retrieval systems.",
            id=document_ids[0],
            domain=["open", "news"],
            topic=topic,
        ),
        Document.from_text(
            "Music and arts commentary about orchestras and pianos in retrieval systems.",
            id=document_ids[1],
            domain=["music", "arts"],
            topic=topic,
        ),
        Document.from_text(
            "Sports and health analysis about football training and recovery metrics.",
            id=document_ids[2],
            domain=["sports", "health"],
            topic=topic,
        ),
    ]
    insert_documents(documents, config=isolated_raglite_test_config)

    try:
        query = "piano orchestra retrieval systems"
        metadata_filter: MetadataFilter = {"topic": topic, "domain": ["open", "music"]}
        chunk_ids, _ = search_method(
            query,
            num_results=10,
            metadata_filter=metadata_filter,
            config=isolated_raglite_test_config,
        )
        assert chunk_ids, "Expected results for list metadata values in domain."

        chunks = retrieve_chunks(chunk_ids, config=isolated_raglite_test_config)
        matched_document_ids = {chunk.document.id for chunk in chunks}
        assert matched_document_ids.issubset(set(document_ids[:2])), (
            "Expected only documents whose domain list overlaps ['open', 'music'], "
            f"got {matched_document_ids}."
        )
        assert matched_document_ids == set(document_ids[:2]), (
            f"Expected both list-based domain matches to be returned, got {matched_document_ids}."
        )
    finally:
        delete_documents(document_ids, config=isolated_raglite_test_config)


def test_self_query_deduplicates_and_keeps_multiple_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep all relevant values in one metadata field for self-query output."""
    from raglite._database import Metadata

    class _Result:
        def model_dump(self, *, exclude_none: bool) -> dict[str, list[int]]:
            assert exclude_none
            return {"domain": [0, 1, 0, 99]}

    monkeypatch.setattr(
        "raglite._search._get_database_metadata",
        lambda **_: [Metadata(name="domain", values=["open", "music", "sports"])],
    )
    monkeypatch.setattr("raglite._search.extract_with_llm", lambda **_: _Result())

    metadata_filter = _self_query("Find open and music results.")
    assert metadata_filter == {"domain": ["open", "music"]}


def test_self_query(raglite_test_config: RAGLiteConfig) -> None:
    """Test self-query functionality that extracts metadata filters from queries."""
    # Test 1: Query that should extract "Physics" from topic field
    query1 = "I want to learn about the topic Physics."
    expected_topic = ["Physics"]
    actual_filter1 = _self_query(query1, config=raglite_test_config)
    assert actual_filter1.get("topic") == expected_topic, (
        f"Expected topic '{expected_topic}', got {actual_filter1.get('topic')}"
    )
    # Test 2: Query with non-existent metadata values should return empty filter
    query2 = "What is the price of a Bugatti Chiron?"
    expected_filter2: dict[str, Any] = {}
    actual_filter2 = _self_query(query2, config=raglite_test_config)
    assert actual_filter2 == expected_filter2, f"Expected {expected_filter2}, got {actual_filter2}"
