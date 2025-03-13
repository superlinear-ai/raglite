"""Test RAGLite's reranking functionality."""

import logging
import random
from typing import TypeVar

import pytest
from rerankers.models.flashrank_ranker import FlashRankRanker
from rerankers.models.ranker import BaseRanker
from scipy.stats import kendalltau

from raglite import RAGLiteConfig, hybrid_search, rerank_chunks, retrieve_chunks
from raglite._database import Chunk

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

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
    chunk_ids, scores = hybrid_search(query, num_results=20, config=raglite_test_config)

    # Log search results
    db_type = "postgres" if "postgres" in raglite_test_config.db_url else "sqlite"
    embedder_type = raglite_test_config.embedder
    reranker_type = (
        "none"
        if reranker is None
        else (
            "flashrank_english"
            if isinstance(reranker, FlashRankRanker)
            else "flashrank_multilingual"
        )
    )

    logging.info(f"Search results for {db_type}-{embedder_type}-{reranker_type}:")
    logging.info(f"  Number of chunk IDs: {len(chunk_ids)}")
    if chunk_ids:
        logging.info(f"  First 5 chunk IDs: {[id[:8] for id in chunk_ids[:5]]}")
        logging.info(f"  First 5 scores: {[f'{score:.4f}' for score in scores[:5]]}")

    # Retrieve the chunks.
    chunks = retrieve_chunks(chunk_ids, config=raglite_test_config)
    logging.info(f"  Number of chunks retrieved: {len(chunks)}")
    if chunks:
        logging.info(f"  First 5 chunk indices: {[chunk.index for chunk in chunks[:5]]}")

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk_id == chunk.id for chunk_id, chunk in zip(chunk_ids, chunks, strict=True))
    # Randomly shuffle the chunks.
    random.seed(42)
    chunks_random = random.sample(chunks, len(chunks))
    # Rerank the chunks starting from a pathological order and verify that it improves the ranking.
    for i, arg in enumerate([chunks[::-1], chunk_ids[::-1]]):
        logging.info(f"Reranking test #{i + 1}: Using {'chunk objects' if i == 0 else 'chunk IDs'}")
        reranked_chunks = rerank_chunks(query, arg, config=raglite_test_config)
        logging.info(f"  Number of reranked chunks: {len(reranked_chunks)}")
        if reranked_chunks:
            logging.info(
                f"  First 5 reranked chunk indices: {[chunk.index for chunk in reranked_chunks[:5]]}"
            )
        if reranker:
            τ_search = kendall_tau(chunks, reranked_chunks)  # noqa: PLC2401
            τ_inverse = kendall_tau(chunks[::-1], reranked_chunks)  # noqa: PLC2401
            τ_random = kendall_tau(chunks_random, reranked_chunks)  # noqa: PLC2401

            # Log detailed information about the chunks and their order
            logging.info(f"Original chunks order (first 5): {[c.index for c in chunks[:5]]}")
            logging.info(f"Random chunks order (first 5): {[c.index for c in chunks_random[:5]]}")
            logging.info(
                f"Reranked chunks order (first 5): {[c.index for c in reranked_chunks[:5]]}"
            )

            # Log the actual content of the first chunk in each ordering
            if chunks:
                logging.info(
                    f"First original chunk content (truncated):\n{chunks[0].body[:100]}..."
                )
            if reranked_chunks:
                logging.info(
                    f"First reranked chunk content (truncated):\n{reranked_chunks[0].body[:100]}..."
                )

            # Log detailed information about the test configuration and results
            db_type = "postgres" if "postgres" in raglite_test_config.db_url else "sqlite"
            embedder_type = raglite_test_config.embedder
            reranker_type = (
                "none"
                if reranker is None
                else (
                    "flashrank_english"
                    if isinstance(reranker, FlashRankRanker)
                    else "flashrank_multilingual"
                )
            )

            logging.info(f"Test configuration: {db_type}-{embedder_type}-{reranker_type}")
            logging.info(
                f"Kendall's Tau values: τ_search={τ_search}, τ_random={τ_random}, τ_inverse={τ_inverse}"
            )

            # Log the first few chunk IDs in each ordering to help diagnose issues
            logging.info(f"Original chunks (first 5): {[c.id[:8] for c in chunks[:5]]}")
            logging.info(f"Random chunks (first 5): {[c.id[:8] for c in chunks_random[:5]]}")
            logging.info(f"Reranked chunks (first 5): {[c.id[:8] for c in reranked_chunks[:5]]}")

            # Check if the assertion will fail and log a warning
            if not (τ_search >= τ_random >= τ_inverse):
                logging.warning(
                    f"ASSERTION WILL FAIL: τ_search={τ_search} >= τ_random={τ_random} >= τ_inverse={τ_inverse}"
                )

            assert τ_search >= τ_random >= τ_inverse
