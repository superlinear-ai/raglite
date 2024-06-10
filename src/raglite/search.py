"""Query documents."""

import json
import re
import string
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from llama_cpp import Llama
from pynndescent import NNDescent
from sqlalchemy.engine import URL
from sqlmodel import Session, select, text

from raglite.database_models import Chunk, ChunkANNIndex, create_database_engine
from raglite.llm import default_llm
from raglite.string_embedder import embed_strings


@lru_cache(maxsize=1)
def _chunk_ann_index(
    index_id: str = "default", db_url: str | URL = "sqlite:///raglite.sqlite"
) -> tuple[NNDescent, np.ndarray]:
    engine = create_database_engine(db_url)
    with Session(engine) as session:
        chunk_ann_index = session.get(ChunkANNIndex, index_id)
        index = chunk_ann_index.index
        chunk_size_cumsum = np.cumsum(np.asarray(chunk_ann_index.chunk_sizes, dtype=np.intp))
        return index, chunk_size_cumsum


def vector_search(
    prompt: str,
    num_results: int = 3,
    index_id: str = "default",
    db_url: str | URL = "sqlite:///raglite.sqlite",
) -> tuple[list[int], list[float]]:
    """Search chunks using ANN vector search."""
    # Retrieve the index from the database.
    index, chunk_size_cumsum = _chunk_ann_index(index_id, db_url)
    # Embed the prompt.
    prompt_embedding = embed_strings([prompt])
    # Find the neighbouring proposition indices.
    proposition_indices, cosine_distance = index.query(prompt_embedding, k=8 * num_results)
    cosine_similarity = 1 - cosine_distance[0, :]
    # Find the neighbouring chunk rowids.
    chunk_offsets = np.searchsorted(chunk_size_cumsum, proposition_indices[0, :], side="right")
    # Remove duplicates.
    _, idx = np.unique(chunk_offsets, return_index=True)
    idx = np.sort(idx)
    chunk_rowids = list(chunk_offsets[idx] + 1)[:num_results]
    cosine_similarity = list(cosine_similarity[idx])[:num_results]
    return chunk_rowids, cosine_similarity


def prompt_to_fts_query(prompt: str) -> str:
    """Convert a prompt to an FTS5 query."""
    # https://www.sqlite.org/fts5.html#full_text_query_syntax
    prompt = re.sub(f"[{re.escape(string.punctuation)}]", "", prompt)
    fts_query = " OR ".join(prompt.split())
    return fts_query


def keyword_search(
    prompt: str,
    num_results: int = 3,
    db_url: str | URL = "sqlite:///raglite.sqlite",
) -> tuple[list[int], list[float]]:
    """Search chunks using BM25 keyword search."""
    engine = create_database_engine(db_url)
    with Session(engine) as session:
        # Perform the full-text search query using the BM25 ranking.
        statement = text(
            "SELECT chunk.rowid, bm25(chunk_fts) FROM chunk JOIN chunk_fts ON chunk.rowid = chunk_fts.rowid WHERE chunk_fts MATCH :match ORDER BY rank LIMIT :limit;"
        )
        results = session.exec(
            statement, params={"match": prompt_to_fts_query(prompt), "limit": num_results}
        )
        # Unpack the results and make FTS5's negative BM25 scores [1] positive.
        # https://www.sqlite.org/fts5.html#the_bm25_function
        chunk_rowids, bm25_score = zip(*results, strict=True)
        chunk_rowids, bm25_score = list(chunk_rowids), [-s for s in bm25_score]
    return chunk_rowids, bm25_score


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> tuple[list[int], list[float]]:
    """Reciprocal Rank Fusion."""
    # Compute the RRF score.
    rowids = {rowid for ranking in rankings for rowid in ranking}
    rowid_score = defaultdict(float)
    for ranking in rankings:
        rowid_index = {rowid: i for i, rowid in enumerate(ranking)}
        for rowid in rowids:
            rowid_score[rowid] += 1 / (k + rowid_index.get(rowid, len(rowid_index)))
    # Rank rrf_results according to descending RRF score.
    rrf_rowids, rrf_score = zip(
        *sorted(rowid_score.items(), key=lambda x: x[1], reverse=True), strict=True
    )
    return rrf_rowids, rrf_score


def hybrid_search(
    prompt: str,
    num_results: int = 3,
    num_rerank: int = 100,
    index_id: str = "default",
    db_url: str | URL = "sqlite:///raglite.sqlite",
) -> tuple[list[int], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search."""
    # Run both searches.
    chunks_vector, _ = vector_search(
        prompt, num_results=num_rerank, index_id=index_id, db_url=db_url
    )
    chunks_bm25, _ = keyword_search(prompt, num_results=num_rerank, db_url=db_url)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_rowids, hybrid_score = reciprocal_rank_fusion([chunks_vector, chunks_bm25])
    chunk_rowids, hybrid_score = chunk_rowids[:num_results], hybrid_score[:num_results]
    return chunk_rowids, hybrid_score


RESPONSE_SCHEMA = {
    "type": "object",
    "description": "An array of questions that help answer the user prompt.",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A single question that helps answer the user prompt.",
        }
    },
    "required": ["questions"],
}

SYSTEM_PROMPT = """
The user will give you a prompt in search of an answer.
Your task is to generate a minimal set of search queries for a search engine that together provide a complete answer to the user prompt.
The search engine expects its search queries to be framed as questions.
ALWAYS format your response according to this JSON schema:
```
{response_schema}
```
""".strip()


def fusion_search(  # noqa: PLR0913
    prompt: str,
    num_results: int = 5,
    num_rerank: int = 100,
    max_tries: int = 4,
    temperature: float = 0.7,
    llm: Callable[[], Llama] = default_llm,
    index_id: str = "default",
    db_url: str | URL = "sqlite:///raglite.sqlite",
) -> tuple[list[int], list[float]]:
    """Search for chunks with the RAG-Fusion method."""
    # Translate the prompt into questions.
    for _ in range(max_tries):
        try:
            # Conver the prompt into a list of questions.
            response = llm().create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(response_schema=RESPONSE_SCHEMA),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object", "schema": RESPONSE_SCHEMA},
                temperature=temperature,
            )
            # Parse the questions from the response.
            questions = json.loads(response["choices"][0]["message"]["content"])["questions"]
            # Basic quality checks.
            if not questions:
                raise ValueError  # noqa: TRY301
            if questions and not all(question[0].isupper() for question in questions):
                raise ValueError  # noqa: TRY301
        except Exception:  # noqa: S112, BLE001, PERF203
            continue
        else:
            questions.append(prompt)
            break
    else:
        questions = [prompt]
    # Collect the search results for all the questions.
    rankings = []
    for question in questions:
        # Run both searches.
        chunks_vector, _ = vector_search(
            question, num_results=num_rerank, index_id=index_id, db_url=db_url
        )
        chunks_bm25, _ = keyword_search(question, num_results=num_rerank, db_url=db_url)
        # Add results to the rankings.
        rankings.append(chunks_vector)
        rankings.append(chunks_bm25)
    # Combine all the search results with Reciprocal Rank Fusion (RRF).
    chunk_rowids, fusion_score = reciprocal_rank_fusion(rankings)
    chunk_rowids, fusion_score = chunk_rowids[:num_results], fusion_score[:num_results]
    return chunk_rowids, fusion_score


def get_chunks(
    chunk_rowids: list[int],
    neighbors: tuple[int, ...] = (-1, 1),
    db_url: str | URL = "sqlite:///raglite.sqlite",
) -> list[str]:
    """Get chunks by id."""
    # Assemble the search results.
    results = []
    engine = create_database_engine(db_url)
    with Session(engine) as session:
        for chunk_rowid in chunk_rowids:
            # Get the relevant chunk at the given index.
            statement = select(Chunk).offset(chunk_rowid - 1)
            chunks = [session.exec(statement).first()]
            # Prepand and append the neighbouring chunks.
            if len(neighbors) > 0:
                document_id = chunks[0].document_id
                index = chunks[0].index
                for offset in sorted(neighbors, key=abs):
                    where = (Chunk.document_id == document_id, Chunk.index == index + offset)
                    chunk = session.exec(select(Chunk).where(*where)).first()
                    if chunk:
                        if offset < 0:
                            chunks.insert(0, chunk)
                        else:
                            chunks.append(chunk)
            # Convert the retrieved chunks to a single string.
            result = chunks[0].headers + "\n\n"
            result += "".join(chunk.body for chunk in chunks)
            results.append(result)
    return results
