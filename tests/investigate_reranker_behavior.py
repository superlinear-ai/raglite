"""Investigate the behavior of the reranker with different inputs."""

import logging
import random
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
from rerankers.models.flashrank_ranker import FlashRankRanker
from scipy.stats import kendalltau

from raglite import RAGLiteConfig, hybrid_search, rerank_chunks, retrieve_chunks
from raglite._database import Chunk, create_database_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_config(db_type: Literal["sqlite", "postgres"], embedder: str) -> RAGLiteConfig:
    """Create a RAGLiteConfig for the given database type and embedder."""
    variant = "local" if embedder.startswith("llama-cpp-python") else "remote"

    if db_type == "postgres":
        db_url = f"postgresql+pg8000://raglite_user:raglite_password@postgres:5432/raglite_test_{variant}"
    else:
        # Use a fixed path for SQLite to ensure we're using the same database
        db_file = Path("/tmp") / f"raglite_test_{variant}.sqlite"
        db_url = f"sqlite:///{db_file}"

    return RAGLiteConfig(db_url=db_url, embedder=embedder)


def get_chunks_for_query(
    query: str, db_type: Literal["sqlite", "postgres"], embedder: str, num_results: int = 20
) -> list[Chunk]:
    """Get chunks for a query from the specified database."""
    config = get_config(db_type, embedder)
    chunk_ids, scores = hybrid_search(query, num_results=num_results, config=config)
    chunks = retrieve_chunks(chunk_ids, config=config)

    logger.info(f"Retrieved {len(chunks)} chunks from {db_type} for query: '{query}'")
    if chunks:
        logger.info(f"First chunk index: {chunks[0].index}")
        logger.info(f"First chunk ID: {chunks[0].id[:8]}")
        logger.info(f"First chunk content (truncated): {chunks[0].body[:100]}...")

    return chunks


def test_reranker_with_same_chunks_different_orders(
    query: str, chunks: list[Chunk], reranker_model: str = "ms-marco-MiniLM-L-12-v2"
) -> None:
    """Test how the reranker behaves with the same chunks in different orders."""
    logger.info("Testing reranker with same chunks in different orders")
    logger.info(f"Number of chunks: {len(chunks)}")

    # Create different orderings of the same chunks
    original_order = chunks.copy()
    reversed_order = chunks.copy()[::-1]
    random.seed(42)
    random_order = random.sample(chunks.copy(), len(chunks))

    # Create a reranker
    reranker = FlashRankRanker(reranker_model, verbose=0)

    # Rerank each ordering
    logger.info("Reranking original order...")
    original_results = reranker.rank(query=query, docs=[str(chunk) for chunk in original_order])

    logger.info("Reranking reversed order...")
    reversed_results = reranker.rank(query=query, docs=[str(chunk) for chunk in reversed_order])

    logger.info("Reranking random order...")
    random_results = reranker.rank(query=query, docs=[str(chunk) for chunk in random_order])

    # Compare the rankings
    original_ranking = [result.doc_id for result in original_results.results]
    reversed_ranking = [result.doc_id for result in reversed_results.results]
    random_ranking = [result.doc_id for result in random_results.results]

    # Check if the rankings are the same
    original_vs_reversed = all(
        o == r for o, r in zip(original_ranking, reversed_ranking, strict=False)
    )
    original_vs_random = all(o == r for o, r in zip(original_ranking, random_ranking, strict=False))

    logger.info(f"Original vs Reversed rankings are identical: {original_vs_reversed}")
    logger.info(f"Original vs Random rankings are identical: {original_vs_random}")

    # Log the first few rankings
    logger.info(f"Original ranking (first 5): {original_ranking[:5]}")
    logger.info(f"Reversed ranking (first 5): {reversed_ranking[:5]}")
    logger.info(f"Random ranking (first 5): {random_ranking[:5]}")

    # Calculate Kendall's Tau between the rankings
    # We need to map the doc_ids back to the original indices
    original_indices = list(range(len(original_order)))
    reversed_indices = list(range(len(reversed_order)))[::-1]
    random_indices = [original_indices.index(random_order.index(chunk)) for chunk in original_order]

    reranked_original_indices = [original_indices[doc_id] for doc_id in original_ranking]
    reranked_reversed_indices = [reversed_indices[doc_id] for doc_id in reversed_ranking]
    reranked_random_indices = [random_indices[doc_id] for doc_id in random_ranking]

    # Calculate Kendall's Tau
    tau_original = kendalltau(original_indices, reranked_original_indices)[0]
    tau_reversed = kendalltau(reversed_indices, reranked_reversed_indices)[0]
    tau_random = kendalltau(random_indices, reranked_random_indices)[0]

    logger.info(f"Kendall's Tau for original order: {tau_original:.4f}")
    logger.info(f"Kendall's Tau for reversed order: {tau_reversed:.4f}")
    logger.info(f"Kendall's Tau for random order: {tau_random:.4f}")

    # Check if the reranker is sensitive to input order
    is_sensitive_to_order = not original_vs_reversed or not original_vs_random
    logger.info(f"Reranker is sensitive to input order: {is_sensitive_to_order}")


def compare_chunk_content(sqlite_chunks: list[Chunk], postgres_chunks: list[Chunk]) -> None:
    """Compare the content of chunks from SQLite and PostgreSQL."""
    logger.info("Comparing chunk content between SQLite and PostgreSQL")
    logger.info(f"SQLite chunks: {len(sqlite_chunks)}")
    logger.info(f"PostgreSQL chunks: {len(postgres_chunks)}")

    # Find common chunk IDs
    sqlite_ids = {chunk.id for chunk in sqlite_chunks}
    postgres_ids = {chunk.id for chunk in postgres_chunks}
    common_ids = sqlite_ids.intersection(postgres_ids)

    logger.info(f"Number of common chunk IDs: {len(common_ids)}")

    if common_ids:
        # Compare content of common chunks
        for chunk_id in common_ids:
            sqlite_chunk = next(chunk for chunk in sqlite_chunks if chunk.id == chunk_id)
            postgres_chunk = next(chunk for chunk in postgres_chunks if chunk.id == chunk_id)

            # Compare basic properties
            same_index = sqlite_chunk.index == postgres_chunk.index
            same_body_length = len(sqlite_chunk.body) == len(postgres_chunk.body)
            same_body = sqlite_chunk.body == postgres_chunk.body

            logger.info(f"Chunk ID: {chunk_id[:8]}")
            logger.info(
                f"  Same index: {same_index} (SQLite: {sqlite_chunk.index}, PostgreSQL: {postgres_chunk.index})"
            )
            logger.info(
                f"  Same body length: {same_body_length} (SQLite: {len(sqlite_chunk.body)}, PostgreSQL: {len(postgres_chunk.body)})"
            )
            logger.info(f"  Same body content: {same_body}")

            if not same_body and same_body_length:
                # Find the first difference
                for i, (s, p) in enumerate(
                    zip(sqlite_chunk.body, postgres_chunk.body, strict=False)
                ):
                    if s != p:
                        logger.info(
                            f"  First difference at position {i}: SQLite '{s}' vs PostgreSQL '{p}'"
                        )
                        break

            # Only check the first few common chunks
            if list(common_ids).index(chunk_id) >= 3:
                break

    # Compare the distribution of chunk indices
    sqlite_indices = [chunk.index for chunk in sqlite_chunks]
    postgres_indices = [chunk.index for chunk in postgres_chunks]

    logger.info(
        f"SQLite chunk indices: min={min(sqlite_indices)}, max={max(sqlite_indices)}, mean={np.mean(sqlite_indices):.2f}"
    )
    logger.info(
        f"PostgreSQL chunk indices: min={min(postgres_indices)}, max={max(postgres_indices)}, mean={np.mean(postgres_indices):.2f}"
    )


def test_reranker_cross_database(
    query: str, sqlite_chunks: list[Chunk], postgres_chunks: list[Chunk]
) -> None:
    """Test the reranker with chunks from different databases."""
    logger.info("Testing reranker with chunks from different databases")

    # Create a reranker
    reranker = FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0)

    # Rerank SQLite chunks
    logger.info("Reranking SQLite chunks...")
    sqlite_results = reranker.rank(query=query, docs=[str(chunk) for chunk in sqlite_chunks])

    # Rerank PostgreSQL chunks
    logger.info("Reranking PostgreSQL chunks...")
    postgres_results = reranker.rank(query=query, docs=[str(chunk) for chunk in postgres_chunks])

    # Get the scores
    sqlite_scores = [result.score for result in sqlite_results.results]
    postgres_scores = [result.score for result in postgres_results.results]

    # Compare score distributions
    sqlite_mean = np.mean(sqlite_scores)
    sqlite_std = np.std(sqlite_scores)
    sqlite_min = min(sqlite_scores)
    sqlite_max = max(sqlite_scores)

    postgres_mean = np.mean(postgres_scores)
    postgres_std = np.std(postgres_scores)
    postgres_min = min(postgres_scores)
    postgres_max = max(postgres_scores)

    logger.info(
        f"SQLite scores: mean={sqlite_mean:.4f}, std={sqlite_std:.4f}, min={sqlite_min:.4f}, max={sqlite_max:.4f}"
    )
    logger.info(
        f"PostgreSQL scores: mean={postgres_mean:.4f}, std={postgres_std:.4f}, min={postgres_min:.4f}, max={postgres_max:.4f}"
    )

    # Check if the score distributions are significantly different
    score_diff = abs(sqlite_mean - postgres_mean)
    logger.info(f"Absolute difference in mean scores: {score_diff:.4f}")
    logger.info(
        f"Relative difference in mean scores: {score_diff / max(sqlite_mean, postgres_mean):.4f}"
    )


def main() -> None:
    """Run the investigation."""
    # Define the query and embedder
    query = "What does it mean for two events to be simultaneous?"
    embedder = "text-embedding-3-small"

    # Get chunks from both databases
    sqlite_chunks = get_chunks_for_query(query, "sqlite", embedder)
    postgres_chunks = get_chunks_for_query(query, "postgres", embedder)

    # Test reranker with same chunks in different orders
    if sqlite_chunks:
        test_reranker_with_same_chunks_different_orders(query, sqlite_chunks)

    if postgres_chunks:
        test_reranker_with_same_chunks_different_orders(query, postgres_chunks)

    # Compare chunk content
    if sqlite_chunks and postgres_chunks:
        compare_chunk_content(sqlite_chunks, postgres_chunks)

    # Test reranker cross-database
    if sqlite_chunks and postgres_chunks:
        test_reranker_cross_database(query, sqlite_chunks, postgres_chunks)


if __name__ == "__main__":
    main()
