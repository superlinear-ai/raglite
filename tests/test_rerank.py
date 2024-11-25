"""Test RAGLite's reranking functionality."""

import pytest
from rerankers.models.flashrank_ranker import FlashRankRanker
from rerankers.models.ranker import BaseRanker

from raglite import RAGLiteConfig, hybrid_search, rerank_chunks, retrieve_chunks
from raglite._database import Chunk


@pytest.fixture(
    params=[
        pytest.param(None, id="no_reranker"),
        pytest.param(FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0), id="flashrank_english"),
        pytest.param(
            (
                ("en", FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0)),
                ("other", FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0)),
            ),
            id="flashrank_multilingual",
        ),
    ],
)
def reranker(
    request: pytest.FixtureRequest,
) -> BaseRanker | tuple[tuple[str, BaseRanker], ...] | None:
    """Get a reranker to test RAGLite with."""
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...] | None = request.param
    return reranker


def test_reranker(
    raglite_test_config: RAGLiteConfig,
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...] | None,
) -> None:
    """Test inserting a document, updating the indexes, and searching for a query."""
    # Update the config with the reranker.
    raglite_test_config = RAGLiteConfig(
        db_url=raglite_test_config.db_url, embedder=raglite_test_config.embedder, reranker=reranker
    )
    # Search for a query.
    query = "What does it mean for two events to be simultaneous?"
    chunk_ids, _ = hybrid_search(query, num_results=10, config=raglite_test_config)
    # Retrieve the chunks.
    chunks = retrieve_chunks(chunk_ids, config=raglite_test_config)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk_id == chunk.id for chunk_id, chunk in zip(chunk_ids, chunks, strict=True))
    # Rerank the chunks given an inverted chunk order.
    reranked_chunks = rerank_chunks(query, chunks[::-1], config=raglite_test_config)
    if reranker is not None and "text-embedding-3-small" not in raglite_test_config.embedder:
        assert reranked_chunks[0] in chunks[:3]
    # Test that we can also rerank given the chunk_ids only.
    reranked_chunks = rerank_chunks(query, chunk_ids[::-1], config=raglite_test_config)
    if reranker is not None and "text-embedding-3-small" not in raglite_test_config.embedder:
        assert reranked_chunks[0] in chunks[:3]
