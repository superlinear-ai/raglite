"""Query documents."""

import re
import string
from collections import defaultdict
from itertools import groupby
from typing import Annotated, ClassVar, cast

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.engine import make_url
from sqlmodel import Session, select, text

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, IndexMetadata, create_database_engine
from raglite._embed import embed_sentences
from raglite._extract import extract_with_llm
from raglite._typing import FloatMatrix


def vector_search(
    prompt: str | FloatMatrix,
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
    # Embed the prompt.
    prompt_embedding = (
        embed_sentences([prompt], config=config)[0, :]
        if isinstance(prompt, str)
        else np.ravel(prompt)
    )
    # Apply the query adapter to the prompt embedding.
    Q = index_metadata.get("query_adapter")  # noqa: N806
    if config.vector_search_query_adapter and Q is not None:
        prompt_embedding = (Q @ prompt_embedding).astype(prompt_embedding.dtype)
    # Search for the multi-vector chunk embeddings that are most similar to the prompt embedding.
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
            distance = distance_func(prompt_embedding).label("distance")
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
            prompt_embedding[np.newaxis, :], k=8 * num_results
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
    prompt: str, *, num_results: int = 3, config: RAGLiteConfig | None = None
) -> tuple[list[str], list[float]]:
    """Search chunks using BM25 keyword search."""
    # Read the config.
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # Connect to the database.
    engine = create_database_engine(config)
    with Session(engine) as session:
        if db_backend == "postgresql":
            # Convert the prompt to a tsquery [1].
            # [1] https://www.postgresql.org/docs/current/textsearch-controls.html
            prompt_escaped = re.sub(r"[&|!():<>\"]", " ", prompt)
            tsv_query = " | ".join(prompt_escaped.split())
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
            # Convert the prompt to an FTS5 query [1].
            # [1] https://www.sqlite.org/fts5.html#full_text_query_syntax
            prompt_escaped = re.sub(f"[{re.escape(string.punctuation)}]", "", prompt)
            fts5_query = " OR ".join(prompt_escaped.split())
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
    prompt: str, *, num_results: int = 3, num_rerank: int = 100, config: RAGLiteConfig | None = None
) -> tuple[list[str], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search."""
    # Run both searches.
    chunkeyword_search_vector, _ = vector_search(prompt, num_results=num_rerank, config=config)
    chunkeyword_search_keyword, _ = keyword_search(prompt, num_results=num_rerank, config=config)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_ids, hybrid_score = reciprocal_rank_fusion(
        [chunkeyword_search_vector, chunkeyword_search_keyword]
    )
    chunk_ids, hybrid_score = chunk_ids[:num_results], hybrid_score[:num_results]
    return chunk_ids, hybrid_score


def fusion_search(
    prompt: str,
    *,
    num_results: int = 5,
    num_rerank: int = 100,
    config: RAGLiteConfig | None = None,
) -> tuple[list[str], list[float]]:
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
        chunkeyword_search_vector, _ = vector_search(query, num_results=num_rerank, config=config)
        chunkeyword_search_keyword, _ = keyword_search(query, num_results=num_rerank, config=config)
        # Add results to the rankings.
        rankings.append(chunkeyword_search_vector)
        rankings.append(chunkeyword_search_keyword)
    # Combine all the search results with Reciprocal Rank Fusion (RRF).
    chunk_ids, fusion_score = reciprocal_rank_fusion(rankings)
    chunk_ids, fusion_score = chunk_ids[:num_results], fusion_score[:num_results]
    return chunk_ids, fusion_score


def retrieve_segments(
    chunk_ids: list[str],
    *,
    neighbors: tuple[int, ...] | None = (-1, 1),
    config: RAGLiteConfig | None = None,
) -> list[str]:
    """Group chunks into contiguous segments and retrieve them."""
    # Get the chunks and extend them with their neighbours.
    config = config or RAGLiteConfig()
    chunks = set()
    engine = create_database_engine(config)
    with Session(engine) as session:
        for chunk_id in chunk_ids:
            # Get the chunk by id.
            chunk = session.get(Chunk, chunk_id)
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
