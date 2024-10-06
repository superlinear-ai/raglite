"""Query documents."""

import re
import string
from collections import defaultdict
from collections.abc import Sequence
from itertools import groupby
from typing import cast

import numpy as np
from langdetect import detect
from sqlalchemy.engine import make_url
from sqlmodel import Session, and_, col, or_, select, text

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, IndexMetadata, create_database_engine
from raglite._embed import embed_sentences
from raglite._typing import FloatMatrix


def vector_search(
    query: str | FloatMatrix,
    *,
    num_results: int = 3,
    config: RAGLiteConfig | None = None,
) -> tuple[list[str], list[float]]:
    """Search chunks using ANN vector search."""
    # Read the config.
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # Get the index metadata (including the query adapter, and in the case of SQLite, the index).
    index_metadata = IndexMetadata.get("default", config=config)
    # Embed the query.
    query_embedding = (
        embed_sentences([query], config=config)[0, :] if isinstance(query, str) else np.ravel(query)
    )
    # Apply the query adapter to the query embedding.
    Q = index_metadata.get("query_adapter")  # noqa: N806
    if config.vector_search_query_adapter and Q is not None:
        query_embedding = (Q @ query_embedding).astype(query_embedding.dtype)
    # Search for the multi-vector chunk embeddings that are most similar to the query embedding.
    if db_backend == "postgresql":
        # Check that the selected metric is supported by pgvector.
        metrics = {"cosine": "<=>", "dot": "<#>", "euclidean": "<->", "l1": "<+>", "l2": "<->"}
        if config.vector_search_index_metric not in metrics:
            error_message = f"Unsupported metric {config.vector_search_index_metric}."
            raise ValueError(error_message)
        # With pgvector, we can obtain the nearest neighbours and similarities with a single query.
        engine = create_database_engine(config)
        with Session(engine) as session:
            distance_func = getattr(
                ChunkEmbedding.embedding, f"{config.vector_search_index_metric}_distance"
            )
            distance = distance_func(query_embedding).label("distance")
            results = session.exec(
                select(ChunkEmbedding.chunk_id, distance).order_by(distance).limit(8 * num_results)
            )
            chunk_ids_, distance = zip(*results, strict=True)
            chunk_ids, similarity = np.asarray(chunk_ids_), 1.0 - np.asarray(distance)
    elif db_backend == "sqlite":
        # Load the NNDescent index.
        index = index_metadata.get("index")
        ids = np.asarray(index_metadata.get("chunk_ids"))
        cumsum = np.cumsum(np.asarray(index_metadata.get("chunk_sizes")))
        # Find the neighbouring multi-vector indices.
        from pynndescent import NNDescent

        multi_vector_indices, distance = cast(NNDescent, index).query(
            query_embedding[np.newaxis, :], k=8 * num_results
        )
        similarity = 1 - distance[0, :]
        # Transform the multi-vector indices into chunk indices, and then to chunk ids.
        chunk_indices = np.searchsorted(cumsum, multi_vector_indices[0, :], side="right") + 1
        chunk_ids = np.asarray([ids[chunk_index - 1] for chunk_index in chunk_indices])
    # Score each unique chunk id as the mean similarity of its multi-vector hits. Chunk ids with
    # fewer hits are padded with the minimum similarity of the result set.
    unique_chunk_ids, counts = np.unique(chunk_ids, return_counts=True)
    score = np.full(
        (len(unique_chunk_ids), np.max(counts)), np.min(similarity), dtype=similarity.dtype
    )
    for i, (unique_chunk_id, count) in enumerate(zip(unique_chunk_ids, counts, strict=True)):
        score[i, :count] = similarity[chunk_ids == unique_chunk_id]
    pooled_similarity = np.mean(score, axis=1)
    # Sort the chunk ids by their adjusted similarity.
    sorted_indices = np.argsort(pooled_similarity)[::-1]
    unique_chunk_ids = unique_chunk_ids[sorted_indices][:num_results]
    pooled_similarity = pooled_similarity[sorted_indices][:num_results]
    return unique_chunk_ids.tolist(), pooled_similarity.tolist()


def keyword_search(
    query: str, *, num_results: int = 3, config: RAGLiteConfig | None = None
) -> tuple[list[str], list[float]]:
    """Search chunks using BM25 keyword search."""
    # Read the config.
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # Connect to the database.
    engine = create_database_engine(config)
    with Session(engine) as session:
        if db_backend == "postgresql":
            # Convert the query to a tsquery [1].
            # [1] https://www.postgresql.org/docs/current/textsearch-controls.html
            query_escaped = re.sub(r"[&|!():<>\"]", " ", query)
            tsv_query = " | ".join(query_escaped.split())
            # Perform keyword search with tsvector.
            statement = text("""
                SELECT id as chunk_id, ts_rank(to_tsvector('simple', body), to_tsquery('simple', :query)) AS score
                FROM chunk
                WHERE to_tsvector('simple', body) @@ to_tsquery('simple', :query)
                ORDER BY score DESC
                LIMIT :limit;
            """)
            results = session.execute(statement, params={"query": tsv_query, "limit": num_results})
        elif db_backend == "sqlite":
            # Convert the query to an FTS5 query [1].
            # [1] https://www.sqlite.org/fts5.html#full_text_query_syntax
            query_escaped = re.sub(f"[{re.escape(string.punctuation)}]", "", query)
            fts5_query = " OR ".join(query_escaped.split())
            # Perform keyword search with FTS5. In FTS5, BM25 scores are negative [1], so we
            # negate them to make them positive.
            # [1] https://www.sqlite.org/fts5.html#the_bm25_function
            statement = text("""
                SELECT chunk.id as chunk_id, -bm25(keyword_search_chunk_index) as score
                FROM chunk JOIN keyword_search_chunk_index ON chunk.rowid = keyword_search_chunk_index.rowid
                WHERE keyword_search_chunk_index MATCH :match
                ORDER BY score DESC
                LIMIT :limit;
            """)
            results = session.execute(statement, params={"match": fts5_query, "limit": num_results})
        # Unpack the results.
        chunk_ids, keyword_score = zip(*results, strict=True)
        chunk_ids, keyword_score = list(chunk_ids), list(keyword_score)  # type: ignore[assignment]
    return chunk_ids, keyword_score  # type: ignore[return-value]


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60
) -> tuple[list[str], list[float]]:
    """Reciprocal Rank Fusion."""
    # Compute the RRF score.
    chunk_ids = {chunk_id for ranking in rankings for chunk_id in ranking}
    chunk_id_score: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        chunk_id_index = {chunk_id: i for i, chunk_id in enumerate(ranking)}
        for chunk_id in chunk_ids:
            chunk_id_score[chunk_id] += 1 / (k + chunk_id_index.get(chunk_id, len(chunk_id_index)))
    # Rank RRF results according to descending RRF score.
    rrf_chunk_ids, rrf_score = zip(
        *sorted(chunk_id_score.items(), key=lambda x: x[1], reverse=True), strict=True
    )
    return list(rrf_chunk_ids), list(rrf_score)


def hybrid_search(
    query: str, *, num_results: int = 3, num_rerank: int = 100, config: RAGLiteConfig | None = None
) -> tuple[list[str], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search."""
    # Run both searches.
    vs_chunk_ids, _ = vector_search(query, num_results=num_rerank, config=config)
    ks_chunk_ids, _ = keyword_search(query, num_results=num_rerank, config=config)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_ids, hybrid_score = reciprocal_rank_fusion([vs_chunk_ids, ks_chunk_ids])
    chunk_ids, hybrid_score = chunk_ids[:num_results], hybrid_score[:num_results]
    return chunk_ids, hybrid_score


def retrieve_chunks(
    chunk_ids: list[str],
    *,
    config: RAGLiteConfig | None = None,
) -> list[Chunk]:
    """Retrieve chunks by their ids."""
    config = config or RAGLiteConfig()
    engine = create_database_engine(config)
    with Session(engine) as session:
        chunks = list(session.exec(select(Chunk).where(col(Chunk.id).in_(chunk_ids))).all())
    return chunks


def retrieve_segments(
    chunk_ids: list[str] | list[Chunk],
    *,
    neighbors: tuple[int, ...] | None = (-1, 1),
    config: RAGLiteConfig | None = None,
) -> list[str]:
    """Group chunks into contiguous segments and retrieve them."""
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Extend the chunks with their neighbouring chunks.
    if neighbors:
        engine = create_database_engine(config)
        with Session(engine) as session:
            neighbor_conditions = [
                and_(Chunk.document_id == chunk.document_id, Chunk.index == chunk.index + offset)
                for chunk in chunks
                for offset in neighbors
            ]
            chunks += list(session.exec(select(Chunk).where(or_(*neighbor_conditions))).all())
    # Keep only the unique chunks.
    chunks = list(set(chunks))
    # Sort the chunks by document_id and index (needed for groupby).
    chunks = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.index))
    # Group the chunks into contiguous segments.
    segments: list[list[Chunk]] = []
    for _, group in groupby(chunks, key=lambda chunk: chunk.document_id):
        segment: list[Chunk] = []
        for chunk in group:
            if not segment or chunk.index == segment[-1].index + 1:
                segment.append(chunk)
            else:
                segments.append(segment)
                segment = [chunk]
        segments.append(segment)
    # Convert the segments into strings.
    segments = [
        segment[0].headings.strip() + "\n\n" + "".join(chunk.body for chunk in segment).strip()  # type: ignore[misc]
        for segment in segments
    ]
    return segments  # type: ignore[return-value]


def rerank(
    query: str,
    chunk_ids: list[str] | list[Chunk],
    *,
    config: RAGLiteConfig | None = None,
) -> list[Chunk]:
    """Rerank chunks according to their relevance to a given query."""
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Early exit if no reranker is configured.
    if not config.reranker:
        return chunks
    # Select the reranker.
    if isinstance(config.reranker, Sequence):
        # Detect the languages of the chunks and queries.
        langs = {detect(str(chunk)) for chunk in chunks}
        langs.add(detect(query))
        # If all chunks and the query are in the same language, use a language-specific reranker.
        rerankers = dict(config.reranker)
        if len(langs) == 1 and (lang := next(iter(langs))) in rerankers:
            reranker = rerankers[lang]
        else:
            reranker = rerankers.get("other")
    else:
        # A specific reranker was configured.
        reranker = config.reranker
    # Rerank the chunks.
    if reranker:
        results = reranker.rank(query=query, docs=[str(chunk) for chunk in chunks])
        chunks = [chunks[result.doc_id] for result in results.results]
    return chunks
