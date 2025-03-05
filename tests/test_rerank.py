"""Test RAGLite's reranking functionality."""

import random
from typing import TypeVar

import pytest
from rerankers.models.flashrank_ranker import FlashRankRanker
from rerankers.models.ranker import BaseRanker
from scipy.stats import kendalltau

from raglite import RAGLiteConfig, hybrid_search, rerank_chunks, retrieve_chunks
from raglite._database import Chunk

T = TypeVar("T")


def kendall_tau(a: list[T], b: list[T]) -> float:
    """Measure the Kendall rank correlation coefficient between two lists."""
    τ: float = kendalltau(range(len(a)), [a.index(el) for el in b])[0]  # noqa: PLC2401
    return τ


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
    chunk_ids, _ = hybrid_search(query, num_results=20, config=raglite_test_config)
    # Retrieve the chunks.
    chunks = retrieve_chunks(chunk_ids, config=raglite_test_config)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk_id == chunk.id for chunk_id, chunk in zip(chunk_ids, chunks, strict=True))
    # Randomly shuffle the chunks.
    random.seed(42)
    chunks_random = random.sample(chunks, len(chunks))
    # Rerank the chunks starting from a pathological order and verify that it improves the ranking.
    for arg in (chunks[::-1], chunk_ids[::-1]):
        reranked_chunks = rerank_chunks(query, arg, config=raglite_test_config)
        if reranker:
            τ_search = kendall_tau(chunks, reranked_chunks)  # noqa: PLC2401
            τ_inverse = kendall_tau(chunks[::-1], reranked_chunks)  # noqa: PLC2401
            τ_random = kendall_tau(chunks_random, reranked_chunks)  # noqa: PLC2401
            # TODO assert that τ_search >= τ_random >= τ_inverse
            assert isinstance(τ_search, float)
            assert isinstance(τ_inverse, float)
            assert isinstance(τ_random, float)
