"""Search and retrieve chunks."""

import contextlib
import re
import string
from collections import defaultdict
from collections.abc import Sequence
from itertools import groupby

import numpy as np
from langdetect import LangDetectException, detect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import joinedload
from sqlmodel import Session, and_, col, func, or_, select, text

from raglite._config import RAGLiteConfig
from raglite._database import (
    Chunk,
    ChunkEmbedding,
    ChunkSpan,
    IndexMetadata,
    create_database_engine,
)
from raglite._embed import embed_strings
from raglite._typing import ChunkId, FloatMatrix


def vector_search(
    query: str | FloatMatrix,
    *,
    num_results: int = 3,
    oversample: int = 4,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks using ANN vector search."""
    # Read the config.
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # Get the index metadata (including the query adapter, and in the case of SQLite, the index).
    index_metadata = IndexMetadata.get("default", config=config)
    # Embed the query.
    query_embedding = (
        embed_strings([query], config=config)[0, :] if isinstance(query, str) else np.ravel(query)
    )
    # Apply the query adapter to the query embedding.
    Q = index_metadata.get("query_adapter")  # noqa: N806
    if config.vector_search_query_adapter and Q is not None:
        query_embedding = (Q @ query_embedding).astype(query_embedding.dtype)
    # Search for the multi-vector chunk embeddings that are most similar to the query embedding.
    num_hits = oversample * max(num_results, 10)
    if db_backend == "postgresql":
        # Rank the chunks by relevance according to the L∞-norm of the similarities of the
        # multi-vector chunk embeddings to the query embedding with a single query.
        engine = create_database_engine(config)
        with Session(engine) as session:
            dist = ChunkEmbedding.embedding.cosine_distance(query_embedding).label("dist")  # type: ignore[attr-defined]
            sim = (1.0 - dist).label("sim")
            top_vectors = (
                select(ChunkEmbedding.chunk_id, sim).order_by(dist).limit(num_hits).subquery()
            )
            sim_norm = func.max(top_vectors.c.sim).label("sim_norm")
            statement = (
                select(top_vectors.c.chunk_id, sim_norm)
                .group_by(top_vectors.c.chunk_id)
                .order_by(sim_norm.desc())
                .limit(num_results)
            )
            rows = session.exec(statement).all()
            chunk_ids = [row[0] for row in rows]
            similarity = [float(row[1]) for row in rows]
    elif db_backend == "sqlite":
        # Load the NNDescent index.
        index = index_metadata.get("index")
        ids = np.asarray(index_metadata.get("chunk_ids", []))
        cumsum = np.cumsum(np.asarray(index_metadata.get("chunk_sizes", [])))
        # Find the neighbouring multi-vector indices.
        from pynndescent import NNDescent

        if isinstance(index, NNDescent) and len(ids) and len(cumsum):
            # Query the index.
            multivector_indices, dist = index.query(
                query_embedding[np.newaxis, :], k=min(num_hits, cumsum[-1])
            )
            # Transform the multi-vector indices into chunk indices.
            chunk_indices = np.searchsorted(cumsum, multivector_indices[0, :], side="right")
            # Compute the L∞-norm of the similarities of the multi-vector chunk embeddings.
            sim_clip = np.maximum(1 - dist[0], 0.0)
            lp_norm = np.zeros(len(ids), dtype=sim_clip.dtype)
            np.maximum.at(lp_norm, chunk_indices, sim_clip)
            # Efficiently find the top chunks.
            num_results = min(num_results, len(ids))
            top_k = np.argpartition(lp_norm, -num_results)[-num_results:]
            top_k = top_k[np.argsort(lp_norm[top_k])[::-1]]
            chunk_ids = [i for i, s in zip(ids[top_k], lp_norm[top_k], strict=True) if s > 0]
            similarity = [float(s) for s in lp_norm[top_k] if s > 0]
        else:
            # Empty result set if there is no index or if no chunks are indexed.
            chunk_ids, similarity = [], []
    return chunk_ids, similarity


def keyword_search(
    query: str, *, num_results: int = 3, config: RAGLiteConfig | None = None
) -> tuple[list[ChunkId], list[float]]:
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
            query_escaped = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)
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
            query_escaped = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)
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
        results = list(results)  # type: ignore[assignment]
        chunk_ids = [result.chunk_id for result in results]
        keyword_score = [result.score for result in results]
    return chunk_ids, keyword_score


def reciprocal_rank_fusion(
    rankings: list[list[ChunkId]], *, k: int = 60, weights: list[float] | None = None
) -> tuple[list[ChunkId], list[float]]:
    """Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        error = "Weights do not match rankings"
        raise ValueError(error)
    # Compute the RRF score.
    chunk_id_score: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=False):
        for i, chunk_id in enumerate(ranking):
            chunk_id_score[chunk_id] += weight / (k + i)
    # Exit early if there are no results to fuse.
    if not chunk_id_score:
        return [], []
    # Rank RRF results according to descending RRF score.
    rrf_chunk_ids, rrf_score = zip(
        *sorted(chunk_id_score.items(), key=lambda x: x[1], reverse=True), strict=True
    )
    return list(rrf_chunk_ids), list(rrf_score)


def hybrid_search(
    query: str, *, num_results: int = 3, oversample: int = 4, config: RAGLiteConfig | None = None
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search."""
    # Run both searches.
    vs_chunk_ids, _ = vector_search(query, num_results=oversample * num_results, config=config)
    ks_chunk_ids, _ = keyword_search(query, num_results=oversample * num_results, config=config)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_ids, hybrid_score = reciprocal_rank_fusion([vs_chunk_ids, ks_chunk_ids])
    chunk_ids, hybrid_score = chunk_ids[:num_results], hybrid_score[:num_results]
    return chunk_ids, hybrid_score


def retrieve_chunks(
    chunk_ids: list[ChunkId], *, config: RAGLiteConfig | None = None
) -> list[Chunk]:
    """Retrieve chunks by their ids."""
    if not chunk_ids:
        return []
    config = config or RAGLiteConfig()
    engine = create_database_engine(config)
    with Session(engine) as session:
        chunks = list(
            session.exec(
                select(Chunk)
                .where(col(Chunk.id).in_(chunk_ids))
                # Eagerly load chunk.document.
                .options(joinedload(Chunk.document))  # type: ignore[arg-type]
            ).all()
        )
    chunks = sorted(chunks, key=lambda chunk: chunk_ids.index(chunk.id))
    return chunks


def rerank_chunks(
    query: str, chunk_ids: list[ChunkId] | list[Chunk], *, config: RAGLiteConfig | None = None
) -> list[Chunk]:
    """Rerank chunks according to their relevance to a given query."""
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, ChunkId) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Exit early if no reranker is configured or if the input is empty.
    if not config.reranker or not chunks:
        return chunks
    # Select the reranker.
    if isinstance(config.reranker, Sequence):
        # Detect the languages of the chunks and queries.
        with contextlib.suppress(LangDetectException):
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


def retrieve_chunk_spans(
    chunk_ids: list[ChunkId] | list[Chunk],
    *,
    neighbors: tuple[int, ...] | None = (-1, 1),
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Group chunks into contiguous chunk spans and retrieve them.

    Chunk spans are ordered according to the aggregate relevance of their underlying chunks, as
    determined by the order in which they are provided to this function.
    """
    # Exit early if the input is empty.
    if not chunk_ids:
        return []
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, ChunkId) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Assign a reciprocal ranking score to each chunk based on its position in the original list.
    chunk_id_to_score = {chunk.id: 1 / (i + 1) for i, chunk in enumerate(chunks)}
    # Extend the chunks with their neighbouring chunks.
    engine = create_database_engine(config)
    with Session(engine) as session:
        if neighbors:
            neighbor_conditions = [
                and_(Chunk.document_id == chunk.document_id, Chunk.index == chunk.index + offset)
                for chunk in chunks
                for offset in neighbors
            ]
            chunks += list(
                session.exec(
                    select(Chunk)
                    .where(or_(*neighbor_conditions))
                    # Eagerly load chunk.document.
                    .options(joinedload(Chunk.document))  # type: ignore[arg-type]
                ).all()
            )
    # Deduplicate and sort the chunks by document_id and index (needed for groupby).
    unique_chunks = sorted(set(chunks), key=lambda chunk: (chunk.document_id, chunk.index))
    # Group the chunks into contiguous segments.
    chunk_spans: list[ChunkSpan] = []
    for _, group in groupby(unique_chunks, key=lambda chunk: chunk.document_id):
        chunk_sequence: list[Chunk] = []
        for chunk in group:
            if not chunk_sequence or chunk.index == chunk_sequence[-1].index + 1:
                chunk_sequence.append(chunk)
            else:
                chunk_spans.append(ChunkSpan(chunks=chunk_sequence))
                chunk_sequence = [chunk]
        chunk_spans.append(ChunkSpan(chunks=chunk_sequence))
    # Rank segments according to the aggregate relevance of their chunks.
    chunk_spans.sort(
        key=lambda chunk_span: sum(
            chunk_id_to_score.get(chunk.id, 0.0) for chunk in chunk_span.chunks
        ),
        reverse=True,
    )
    return chunk_spans
