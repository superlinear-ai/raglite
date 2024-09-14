"""Test RAGLite's basic functionality."""

from pathlib import Path

from raglite import RAGLiteConfig, hybrid_search, insert_document, retrieve_segments


def test_insert_index_search(simple_config: RAGLiteConfig) -> None:
    """Test inserting a document, updating the vector index, and searching for a query."""
    # Insert a document and update the index.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=simple_config)
    # Search for a query.
    query = "What does it mean for two events to be simultaneous?"
    chunk_ids, scores = hybrid_search(query, config=simple_config)
    assert len(chunk_ids) == len(scores)
    assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
    assert all(isinstance(score, float) for score in scores)
    # Group the chunks into segments and retrieve them.
    segments = retrieve_segments(chunk_ids, neighbors=None, config=simple_config)
    assert all(isinstance(segment, str) for segment in segments)
    assert "Definition of Simultaneity" in "".join(segments[:2])
