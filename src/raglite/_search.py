"""Query documents."""

import re
import string
from collections import defaultdict
from functools import lru_cache
from itertools import groupby
from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field
from pynndescent import NNDescent
from sqlmodel import Session, select, text

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, VectorSearchChunkIndex, create_database_engine
from raglite._embed import embed_strings
from raglite._extract import extract_with_llm
from raglite._typing import FloatMatrix, IntVector


@lru_cache(maxsize=1)
def _vector_search_chunk_index(
    config: RAGLiteConfig,
) -> tuple[NNDescent, IntVector, FloatMatrix | None]:
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        vector_search_chunk_index = session.get(
            VectorSearchChunkIndex, config.vector_search_index_id
        )
        if vector_search_chunk_index is None:
            error_message = "First run `update_vector_index()` to create a vector search index."
            raise ValueError(error_message)
        index = vector_search_chunk_index.index
        chunk_size_cumsum = np.cumsum(
            np.asarray(vector_search_chunk_index.chunk_sizes, dtype=np.intp)
        )
        query_adapter = vector_search_chunk_index.query_adapter
    return index, chunk_size_cumsum, query_adapter


def vector_search(
    prompt: str | FloatMatrix,
    *,
    num_results: int = 3,
    query_adapter: bool = True,
    config: RAGLiteConfig | None = None,
) -> tuple[list[int], list[float]]:
    """Search chunks using ANN vector search."""
    # Retrieve the index from the database.
    config = config or RAGLiteConfig()
    index, chunk_size_cumsum, Q = _vector_search_chunk_index(config)  # noqa: N806
    # Embed the prompt.
    prompt_embedding = (
        embed_strings([prompt], config=config)
        if isinstance(prompt, str)
        else np.reshape(prompt, (1, -1))
    )
    # Apply the query adapter.
    if query_adapter and Q is not None:
        prompt_embedding = (Q @ prompt_embedding[0, :])[np.newaxis, :].astype(config.embedder_dtype)
    # Find the neighbouring multi-vector indices.
    multi_vector_indices, cosine_distance = index.query(prompt_embedding, k=8 * num_results)
    cosine_similarity = 1 - cosine_distance[0, :]
    # Transform the multi-vector indices into chunk rowids.
    chunk_rowids = np.searchsorted(chunk_size_cumsum, multi_vector_indices[0, :], side="right") + 1
    # Score each unique chunk rowid as the mean cosine similarity of its multi-vector hits.
    # Chunk rowids with fewer hits are padded with the minimum cosine similarity of the result set.
    unique_chunk_rowids, counts = np.unique(chunk_rowids, return_counts=True)
    score = np.full(
        (len(unique_chunk_rowids), np.max(counts)),
        np.min(cosine_similarity),
        dtype=cosine_similarity.dtype,
    )
    for i, (unique_chunk_rowid, count) in enumerate(zip(unique_chunk_rowids, counts, strict=True)):
        score[i, :count] = cosine_similarity[chunk_rowids == unique_chunk_rowid]
    pooled_cosine_similarity = np.mean(score, axis=1)
    # Sort the chunk rowids by adjusted cosine similarity.
    sorted_indices = np.argsort(pooled_cosine_similarity)[::-1]
    unique_chunk_rowids = unique_chunk_rowids[sorted_indices][:num_results]
    pooled_cosine_similarity = pooled_cosine_similarity[sorted_indices][:num_results]
    return list(unique_chunk_rowids), list(pooled_cosine_similarity)


def _prompt_to_fts_query(prompt: str) -> str:
    """Convert a prompt to an FTS5 query."""
    # https://www.sqlite.org/fts5.html#full_text_query_syntax
    prompt = re.sub(f"[{re.escape(string.punctuation)}]", "", prompt)
    fts_query = " OR ".join(prompt.split())
    return fts_query


def keyword_search(
    prompt: str, *, num_results: int = 3, config: RAGLiteConfig | None = None
) -> tuple[list[int], list[float]]:
    """Search chunks using BM25 keyword search."""
    config = config or RAGLiteConfig()
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        # Perform the full-text search query using the BM25 ranking.
        statement = text(
            "SELECT chunk.rowid, bm25(fts_chunk_index) FROM chunk JOIN fts_chunk_index ON chunk.rowid = fts_chunk_index.rowid WHERE fts_chunk_index MATCH :match ORDER BY rank LIMIT :limit;"
        )
        results = session.execute(
            statement, params={"match": _prompt_to_fts_query(prompt), "limit": num_results}
        )
        # Unpack the results and make FTS5's negative BM25 scores [1] positive.
        # https://www.sqlite.org/fts5.html#the_bm25_function
        chunk_rowids, bm25_score = zip(*results, strict=True)
        chunk_rowids, bm25_score = list(chunk_rowids), [-s for s in bm25_score]  # type: ignore[assignment]
    return chunk_rowids, bm25_score  # type: ignore[return-value]


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, k: int = 60
) -> tuple[list[int], list[float]]:
    """Reciprocal Rank Fusion."""
    # Compute the RRF score.
    rowids = {rowid for ranking in rankings for rowid in ranking}
    rowid_score: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        rowid_index = {rowid: i for i, rowid in enumerate(ranking)}
        for rowid in rowids:
            rowid_score[rowid] += 1 / (k + rowid_index.get(rowid, len(rowid_index)))
    # Rank RRF results according to descending RRF score.
    rrf_rowids, rrf_score = zip(
        *sorted(rowid_score.items(), key=lambda x: x[1], reverse=True), strict=True
    )
    return list(rrf_rowids), list(rrf_score)


def hybrid_search(
    prompt: str, *, num_results: int = 3, num_rerank: int = 100, config: RAGLiteConfig | None = None
) -> tuple[list[int], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search."""
    # Run both searches.
    chunks_vector, _ = vector_search(prompt, num_results=num_rerank, config=config)
    chunks_bm25, _ = keyword_search(prompt, num_results=num_rerank, config=config)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_rowids, hybrid_score = reciprocal_rank_fusion([chunks_vector, chunks_bm25])
    chunk_rowids, hybrid_score = chunk_rowids[:num_results], hybrid_score[:num_results]
    return chunk_rowids, hybrid_score


def fusion_search(
    prompt: str,
    *,
    num_results: int = 5,
    num_rerank: int = 100,
    config: RAGLiteConfig | None = None,
) -> tuple[list[int], list[float]]:
    """Search for chunks with the RAG-Fusion method."""

    class QueriesResponse(BaseModel):
        """An array of queries that help answer the user prompt."""

        queries: list[Annotated[str, Field(min_length=1)]] = Field(
            ..., description="A single query that helps answer the user prompt."
        )
        system_prompt: ClassVar[str] = """
The user will give you a prompt in search of an answer.
Your task is to generate a minimal set of search queries for a search engine that together provide a complete answer to the user prompt.
            """.strip()

    try:
        queries_response = extract_with_llm(QueriesResponse, prompt, config=config)
    except ValueError:
        queries = [prompt]
    else:
        queries = [*queries_response.queries, prompt]
    # Collect the search results for all the queries.
    rankings = []
    for query in queries:
        # Run both searches.
        chunks_vector, _ = vector_search(query, num_results=num_rerank, config=config)
        chunks_bm25, _ = keyword_search(query, num_results=num_rerank, config=config)
        # Add results to the rankings.
        rankings.append(chunks_vector)
        rankings.append(chunks_bm25)
    # Combine all the search results with Reciprocal Rank Fusion (RRF).
    chunk_rowids, fusion_score = reciprocal_rank_fusion(rankings)
    chunk_rowids, fusion_score = chunk_rowids[:num_results], fusion_score[:num_results]
    return chunk_rowids, fusion_score


def retrieve_segments(
    chunk_rowids: list[int],
    *,
    neighbors: tuple[int, ...] | None = (-1, 1),
    config: RAGLiteConfig | None = None,
) -> list[str]:
    """Group the chunks into contiguous segments and retrieve them."""
    # Get the chunks by rowid and extend them with their neighbours.
    config = config or RAGLiteConfig()
    chunks = set()
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        for chunk_rowid in chunk_rowids:
            # Get the chunk at the given rowid.
            chunk = session.exec(select(Chunk).offset(chunk_rowid - 1)).first()
            if chunk is not None:
                chunks.add(chunk)
            # Extend the chunk with its neighbouring chunks.
            if chunk is not None and neighbors is not None and len(neighbors) > 0:
                for offset in sorted(neighbors, key=abs):
                    where = (
                        Chunk.document_id == chunk.document_id,
                        Chunk.index == chunk.index + offset,
                    )
                    neighbor = session.exec(select(Chunk).where(*where)).first()
                    if neighbor is not None:
                        chunks.add(neighbor)
    # Sort the chunks by document_id and index (needed for groupby).
    chunks = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.index))  # type: ignore[assignment]
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
