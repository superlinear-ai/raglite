"""Investigate the reranker issue with PostgreSQL + OpenAI + FlashRank English."""

import logging
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rerankers.models.flashrank_ranker import FlashRankRanker
from scipy.stats import kendalltau
from sqlalchemy.engine import make_url
from sqlalchemy.sql import text
from sqlmodel import Session, select

from raglite import RAGLiteConfig, hybrid_search, rerank_chunks, retrieve_chunks
from raglite._database import Chunk, ChunkEmbedding, create_database_engine
from raglite._search import vector_search

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def kendall_tau(a: list[Any], b: list[Any]) -> float:
    """Measure the Kendall rank correlation coefficient between two lists."""
    if not a or not b or len(a) != len(b):
        logger.warning(f"Invalid lists for Kendall's Tau: a={len(a)}, b={len(b)}")
        return float("nan")

    try:
        # Check if all elements in b are in a
        if not all(el in a for el in b):
            logger.warning("Not all elements in b are in a")
            return float("nan")

        τ: float = kendalltau(range(len(a)), [a.index(el) for el in b])[0]
        return τ
    except Exception as e:
        logger.warning(f"Error calculating Kendall's Tau: {e}")
        return float("nan")


def get_config(db_type: Literal["sqlite", "postgres"], embedder: str) -> RAGLiteConfig:
    """Create a RAGLiteConfig for the given database type and embedder."""
    variant = "local" if embedder.startswith("llama-cpp-python") else "remote"

    if db_type == "postgres":
        db_url = f"postgresql+pg8000://raglite_user:raglite_password@postgres:5432/raglite_test_{variant}"
    else:
        # Use a fixed path for SQLite to ensure we're using the same database
        db_file = Path("/tmp") / f"raglite_test_{variant}.sqlite"
        db_url = f"sqlite:///{db_file}"

    return RAGLiteConfig(
        db_url=db_url,
        embedder=embedder,
        reranker=FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0),
    )


def compare_search_results(query: str, embedder: str) -> None:
    """Compare search results between SQLite and PostgreSQL for the same query."""
    logger.info(f"Comparing search results for query: '{query}' with embedder: {embedder}")

    # Get configs for both database types
    sqlite_config = get_config("sqlite", embedder)
    postgres_config = get_config("postgres", embedder)

    # Run hybrid search on both databases
    sqlite_chunk_ids, sqlite_scores = hybrid_search(query, num_results=20, config=sqlite_config)
    postgres_chunk_ids, postgres_scores = hybrid_search(
        query, num_results=20, config=postgres_config
    )

    # Retrieve chunks
    sqlite_chunks = retrieve_chunks(sqlite_chunk_ids, config=sqlite_config)
    postgres_chunks = retrieve_chunks(postgres_chunk_ids, config=postgres_config)

    # Compare results
    common_ids = set(sqlite_chunk_ids).intersection(set(postgres_chunk_ids))
    logger.info(f"Number of common chunk IDs: {len(common_ids)} out of 20")

    # Calculate Kendall's Tau between the two result sets
    # We need to find the indices of common IDs in both result sets
    common_sqlite_indices = [sqlite_chunk_ids.index(chunk_id) for chunk_id in common_ids]
    common_postgres_indices = [postgres_chunk_ids.index(chunk_id) for chunk_id in common_ids]

    if common_ids:
        tau = kendalltau(common_sqlite_indices, common_postgres_indices)[0]
        logger.info(f"Kendall's Tau between SQLite and PostgreSQL results: {tau}")

    # Log the first few results from each database
    logger.info("SQLite results (first 5):")
    for i, (chunk_id, score) in enumerate(zip(sqlite_chunk_ids[:5], sqlite_scores[:5])):
        logger.info(f"  #{i}: ID={chunk_id[:8]}, Score={score:.4f}")

    logger.info("PostgreSQL results (first 5):")
    for i, (chunk_id, score) in enumerate(zip(postgres_chunk_ids[:5], postgres_scores[:5])):
        logger.info(f"  #{i}: ID={chunk_id[:8]}, Score={score:.4f}")


def examine_embeddings(embedder: str) -> None:
    """Examine embedding vectors in both databases."""
    logger.info(f"Examining embeddings for embedder: {embedder}")

    # Get configs for both database types
    sqlite_config = get_config("sqlite", embedder)
    postgres_config = get_config("postgres", embedder)

    # Create database engines
    sqlite_engine = create_database_engine(sqlite_config)
    postgres_engine = create_database_engine(postgres_config)

    # Sample a few chunk IDs to examine
    with Session(sqlite_engine) as session:
        sample_chunks = session.exec(select(Chunk).limit(5)).all()
        sample_chunk_ids = [chunk.id for chunk in sample_chunks]

    # Examine embeddings for these chunks in both databases
    for chunk_id in sample_chunk_ids:
        logger.info(f"Examining embeddings for chunk ID: {chunk_id[:8]}")

        # Get embeddings from SQLite
        with Session(sqlite_engine) as session:
            sqlite_embeddings = session.exec(
                select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == chunk_id)
            ).all()

            if sqlite_embeddings:
                sqlite_embedding = sqlite_embeddings[0].embedding
                logger.info(
                    f"SQLite embedding: shape={len(sqlite_embedding)}, "
                    f"min={min(sqlite_embedding):.4f}, max={max(sqlite_embedding):.4f}, "
                    f"mean={np.mean(sqlite_embedding):.4f}, std={np.std(sqlite_embedding):.4f}"
                )

        # Get embeddings from PostgreSQL
        with Session(postgres_engine) as session:
            postgres_embeddings = session.exec(
                select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == chunk_id)
            ).all()

            if postgres_embeddings:
                postgres_embedding = postgres_embeddings[0].embedding
                logger.info(
                    f"PostgreSQL embedding: shape={len(postgres_embedding)}, "
                    f"min={min(postgres_embedding):.4f}, max={max(postgres_embedding):.4f}, "
                    f"mean={np.mean(postgres_embedding):.4f}, std={np.std(postgres_embedding):.4f}"
                )

                # Compare embeddings
                if sqlite_embeddings:
                    similarity = np.dot(sqlite_embedding, postgres_embedding) / (
                        np.linalg.norm(sqlite_embedding) * np.linalg.norm(postgres_embedding)
                    )
                    logger.info(
                        f"Cosine similarity between SQLite and PostgreSQL embeddings: {similarity:.4f}"
                    )


def test_reranker_directly(query: str, embedder: str) -> None:
    """Test the reranker directly with controlled inputs."""
    logger.info(f"Testing reranker directly for query: '{query}' with embedder: {embedder}")

    # Get configs for both database types
    sqlite_config = get_config("sqlite", embedder)
    postgres_config = get_config("postgres", embedder)

    # Run vector search on both databases to get initial results
    sqlite_chunk_ids, _ = vector_search(query, num_results=20, config=sqlite_config)
    postgres_chunk_ids, _ = vector_search(query, num_results=20, config=postgres_config)

    # Retrieve chunks
    sqlite_chunks = retrieve_chunks(sqlite_chunk_ids, config=sqlite_config)
    postgres_chunks = retrieve_chunks(postgres_chunk_ids, config=postgres_config)

    # Log chunk information
    logger.info(f"SQLite chunks: {len(sqlite_chunks)}")
    logger.info(f"PostgreSQL chunks: {len(postgres_chunks)}")

    if not sqlite_chunks or not postgres_chunks:
        logger.warning("No chunks found in one or both databases")
        return

    # Create a reranker
    reranker = FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0)

    # Test reranker with SQLite chunks
    logger.info("Testing reranker with SQLite chunks")
    random.seed(42)
    sqlite_chunks_random = random.sample(sqlite_chunks, len(sqlite_chunks))

    for i, arg in enumerate([sqlite_chunks[::-1], sqlite_chunk_ids[::-1]]):
        logger.info(f"SQLite test #{i + 1}: Using {'chunk objects' if i == 0 else 'chunk IDs'}")
        sqlite_reranked_chunks = rerank_chunks(query, arg, config=sqlite_config)
        logger.info(f"Reranked chunks: {len(sqlite_reranked_chunks)}")

        # Check if reranking worked
        if not sqlite_reranked_chunks:
            logger.warning("Reranking returned no chunks")
            continue

        # Check if all original chunks are in reranked chunks
        missing = set(c.id for c in sqlite_chunks) - set(c.id for c in sqlite_reranked_chunks)
        if missing:
            logger.warning(f"Some chunks are missing after reranking: {len(missing)}")

        τ_search = kendall_tau(sqlite_chunks, sqlite_reranked_chunks)
        τ_inverse = kendall_tau(sqlite_chunks[::-1], sqlite_reranked_chunks)
        τ_random = kendall_tau(sqlite_chunks_random, sqlite_reranked_chunks)

        logger.info(
            f"SQLite Kendall's Tau values: τ_search={τ_search:.4f}, τ_random={τ_random:.4f}, τ_inverse={τ_inverse:.4f}"
        )
        logger.info(f"SQLite assertion: {τ_search >= τ_random >= τ_inverse}")

    # Test reranker with PostgreSQL chunks
    logger.info("Testing reranker with PostgreSQL chunks")
    random.seed(42)
    postgres_chunks_random = random.sample(postgres_chunks, len(postgres_chunks))

    for i, arg in enumerate([postgres_chunks[::-1], postgres_chunk_ids[::-1]]):
        logger.info(f"PostgreSQL test #{i + 1}: Using {'chunk objects' if i == 0 else 'chunk IDs'}")
        postgres_reranked_chunks = rerank_chunks(query, arg, config=postgres_config)
        logger.info(f"Reranked chunks: {len(postgres_reranked_chunks)}")

        # Check if reranking worked
        if not postgres_reranked_chunks:
            logger.warning("Reranking returned no chunks")
            continue

        # Check if all original chunks are in reranked chunks
        missing = set(c.id for c in postgres_chunks) - set(c.id for c in postgres_reranked_chunks)
        if missing:
            logger.warning(f"Some chunks are missing after reranking: {len(missing)}")

        τ_search = kendall_tau(postgres_chunks, postgres_reranked_chunks)
        τ_inverse = kendall_tau(postgres_chunks[::-1], postgres_reranked_chunks)
        τ_random = kendall_tau(postgres_chunks_random, postgres_reranked_chunks)

        logger.info(
            f"PostgreSQL Kendall's Tau values: τ_search={τ_search:.4f}, τ_random={τ_random:.4f}, τ_inverse={τ_inverse:.4f}"
        )
        logger.info(f"PostgreSQL assertion: {τ_search >= τ_random >= τ_inverse}")

    # Test reranker with SQLite chunks but PostgreSQL config
    logger.info("Testing reranker with SQLite chunks but PostgreSQL config")
    sqlite_reranked_chunks = rerank_chunks(query, sqlite_chunks[::-1], config=postgres_config)
    logger.info(f"Mixed test reranked chunks: {len(sqlite_reranked_chunks)}")

    # Check if reranking worked
    if not sqlite_reranked_chunks:
        logger.warning("Mixed test reranking returned no chunks")
        return

    # Check if all original chunks are in reranked chunks
    missing = set(c.id for c in sqlite_chunks) - set(c.id for c in sqlite_reranked_chunks)
    if missing:
        logger.warning(f"Some chunks are missing after mixed test reranking: {len(missing)}")

    τ_search = kendall_tau(sqlite_chunks, sqlite_reranked_chunks)
    τ_inverse = kendall_tau(sqlite_chunks[::-1], sqlite_reranked_chunks)
    τ_random = kendall_tau(sqlite_chunks_random, sqlite_reranked_chunks)

    logger.info(
        f"Mixed Kendall's Tau values: τ_search={τ_search:.4f}, τ_random={τ_random:.4f}, τ_inverse={τ_inverse:.4f}"
    )
    logger.info(f"Mixed assertion: {τ_search >= τ_random >= τ_inverse}")


def investigate_pgvector_config(embedder: str) -> None:
    """Investigate pgvector configuration."""
    logger.info(f"Investigating pgvector configuration for embedder: {embedder}")

    # Get PostgreSQL config
    postgres_config = get_config("postgres", embedder)

    # Create database engine
    postgres_engine = create_database_engine(postgres_config)

    # Check pgvector configuration
    with Session(postgres_engine) as session:
        # Check vector index
        result = session.execute(
            text("SELECT * FROM pg_indexes WHERE indexname = 'vector_search_chunk_index'")
        )
        indexes = list(result)
        logger.info(f"Vector index: {indexes}")

        # Check pgvector version
        result = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        version = result.scalar_one()
        logger.info(f"pgvector version: {version}")

        # Check HNSW parameters
        result = session.execute(text("SHOW hnsw.ef_search"))
        ef_search = result.scalar_one()
        logger.info(f"hnsw.ef_search: {ef_search}")

        result = session.execute(text("SHOW hnsw.iterative_scan"))
        iterative_scan = result.scalar_one()
        logger.info(f"hnsw.iterative_scan: {iterative_scan}")


def main() -> None:
    """Run the investigation."""
    # Define the query and embedder
    query = "What does it mean for two events to be simultaneous?"
    embedder = "text-embedding-3-small"

    # Run the investigations
    compare_search_results(query, embedder)
    examine_embeddings(embedder)
    test_reranker_directly(query, embedder)
    investigate_pgvector_config(embedder)


if __name__ == "__main__":
    main()
