"""Test RAGLite's basic functionality."""

from pathlib import Path

from raglite import (
    RAGLiteConfig,
    hybrid_search,
    insert_document,
    retrieve_segments,
    update_vector_index,
)


def test_insert_index_search() -> None:
    """Test inserting a document, updating the vector index, and searching for a query."""
    # Run this test with an in-memory SQLite database.
    in_memory_db = RAGLiteConfig(db_url="sqlite:///:memory:")
    # Insert a document.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=in_memory_db)
    # Update the vector index with the new document.
    update_vector_index(config=in_memory_db)
    # Search for a query.
    query = "What does it mean for two events to be simultaneous?"
    chunk_rowids, scores = hybrid_search(query, config=in_memory_db)
    assert len(chunk_rowids) == len(scores)
    assert all(isinstance(rowid, int) for rowid in chunk_rowids)
    assert all(isinstance(score, float) for score in scores)
    # Group the chunks into segments and retrieve them.
    segments = retrieve_segments(chunk_rowids, neighbors=None, config=in_memory_db)
    assert all(isinstance(segment, str) for segment in segments)
    assert "Definition of Simultaneity" in segments[0]
