"""Query adapter."""

from collections.abc import Callable
from random import randint
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Eval, VectorSearchChunkIndex, create_database_engine
from raglite._embed import embed_strings
from raglite._extract import extract_with_llm
from raglite._rag import rag
from raglite._search import hybrid_search, retrieve_segments, vector_search


def insert_evals(  # noqa: C901
    *, num_evals: int = 100, max_contexts_per_eval: int = 20, config: RAGLiteConfig | None = None
) -> None:
    """Generate and insert evals into the database."""

    class QuestionResponse(BaseModel):
        """A specific question about the content of a set of document contexts."""

        question: str = Field(
            ...,
            description="A specific question about the content of a set of document contexts.",
            min_length=1,
        )
        system_prompt: ClassVar[str] = """
You are given a set of contexts extracted from a document.
You are a subject matter expert on the document's topic.
Your task is to generate a question to quiz other subject matter experts on the information in the provided context.
The question MUST satisfy ALL of the following criteria:
- The question SHOULD integrate as much of the provided context as possible.
- The question MUST NOT be a general or open question, but MUST instead be as specific to the provided context as possible.
- The question MUST be completely answerable using ONLY the information in the provided context, without depending on any background information.
- The question MUST be entirely self-contained and able to be understood in full WITHOUT access to the provided context.
- The question MUST NOT use the words "context", "contexts", "document", or "question".
            """.strip()

        @field_validator("question")
        @classmethod
        def validate_question(cls, value: str) -> str:
            """Validate the question."""
            question = value.strip().lower()
            if "context" in question or "document" in question or "question" in question:
                raise ValueError
            if not question.endswith("?"):
                raise ValueError
            if not value[0].isupper():
                raise ValueError
            return value

    config = config or RAGLiteConfig()
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        # Sample random chunks from the database.
        seed_chunks = session.exec(select(Chunk).order_by(Chunk.id).limit(num_evals)).all()
        for seed_chunk in tqdm(
            seed_chunks, desc="Generating evals", unit="eval", dynamic_ncols=True
        ):
            # Expand the seed chunk into a set of related chunks.
            related_chunk_rowids, _ = vector_search(
                np.mean(seed_chunk.multi_vector_embedding, axis=0, keepdims=True),
                num_results=randint(2, max_contexts_per_eval),  # noqa: S311
                config=config,
            )
            related_chunks = retrieve_segments(related_chunk_rowids, config=config)
            # Extract a question from the seed chunk's related chunks.
            try:
                question_response = extract_with_llm(
                    QuestionResponse, related_chunks, config=config
                )
            except ValueError:
                continue
            else:
                question = question_response.question
            # Search for candidate chunks to answer the generated question.
            candidate_chunk_rowids, _ = hybrid_search(
                question, num_results=max_contexts_per_eval, config=config
            )
            candidate_chunks = [
                session.exec(select(Chunk).offset(chunk_rowid - 1)).first()
                for chunk_rowid in candidate_chunk_rowids
            ]

            # Determine which candidate chunks are relevant to answer the generated question.
            class ContextEvalResponse(BaseModel):
                """Indicate whether the provided context can be used to answer a given question."""

                hit: bool = Field(
                    ...,
                    description="True if the provided context contains (a part of) the answer to the given question, false otherwise.",
                )
                system_prompt: ClassVar[str] = f"""
You are given a context extracted from a document.
You are a subject matter expert on the document's topic.
Your task is to answer whether the provided context contains (a part of) the answer to this question: "{question}"
An example of a context that does NOT contain (a part of) the answer is a table of contents.
                    """.strip()

            relevant_chunks = []
            for candidate_chunk in tqdm(
                candidate_chunks, desc="Evaluating chunks", unit="chunk", dynamic_ncols=True
            ):
                try:
                    context_eval_response = extract_with_llm(
                        ContextEvalResponse, str(candidate_chunk), config=config
                    )
                except ValueError:  # noqa: PERF203
                    pass
                else:
                    if context_eval_response.hit:
                        relevant_chunks.append(candidate_chunk)

            # Answer the question using the relevant chunks.
            class AnswerResponse(BaseModel):
                """Answer a question using the provided context."""

                answer: str = Field(
                    ...,
                    description="A complete answer to the given question using the provided context.",
                    min_length=1,
                )
                system_prompt: ClassVar[str] = f"""
You are given a set of contexts extracted from a document.
You are a subject matter expert on the document's topic.
Your task is to generate a complete answer to the following question using the provided context: "{question}"
The answer MUST satisfy ALL of the following criteria:
- The answer MUST integrate as much of the provided context as possible.
- The answer MUST be entirely self-contained and able to be understood in full WITHOUT access to the provided context.
- The answer MUST NOT use the words "context", "contexts", or "document".
                    """.strip()

            try:
                answer_response = extract_with_llm(
                    AnswerResponse,
                    [str(relevant_chunk) for relevant_chunk in relevant_chunks],
                    config=config,
                )
            except ValueError:
                continue
            else:
                answer = answer_response.answer
            # Store the eval in the database.
            eval_ = Eval.from_chunks(
                question=question,
                contexts=relevant_chunks,
                ground_truth=answer,
            )
            session.add(eval_)
            session.commit()


def update_query_adapter(  # noqa: C901
    *,
    max_triplets: int = 4096,
    max_triplets_per_eval: int = 64,
    optimize_top_k: int = 40,
    config: RAGLiteConfig | None = None,
) -> None:
    """Compute an optimal query adapter and update the database with it.

    Computes an optimal linear transform A, called a 'query adapter', that is used to transform
    a query embedding q as Aq before searching for the nearest neighbouring chunks in order to
    improve the quality of the search results.

    Given a set of triplets (qᵢ, pᵢ, nᵢ), we want to find the query adapter A that increases the
    score pᵢ'qᵢ of the positive chunk pᵢ and decreases the score nᵢ'qᵢ of the negative chunk nᵢ.

    If the nearest neighbour search uses the dot product as its ranking function, we can find the
    optimal query adapter by solving the following relaxed Procrustes optimisation problem with a
    bound on the Frobenius norm of A:

    A* = argmax Σᵢ pᵢ' (A qᵢ) - nᵢ' (A qᵢ)
                Σᵢ (pᵢ - nᵢ)' A qᵢ
                trace[ (P - N) A Q' ]  where  Q := [q₁'; ...; qₖ']
                                              P := [p₁'; ...; pₖ']
                                              N := [n₁'; ...; nₖ']
                trace[ Q' (P - N) A ]
                trace[ M A ]           where  M := Q' (P - N)
           s.t. ||A||_F == 1
       = M' / ||M||_F

    If the nearest neighbour search uses the cosine similarity as its ranking function, we can find
    the optimal query adapter by solving the following orthogonal Procrustes optimisation problem
    with an orthogonality constraint on A:

    A* = argmax Σᵢ pᵢ' (A qᵢ) - nᵢ' (A qᵢ)
                Σᵢ (pᵢ - nᵢ)' A qᵢ
                trace[ (P - N) A Q' ]
                trace[ Q' (P - N) A ]
                trace[ M A ]
                trace[ U Σ V' A ]      where  U Σ V' := M is the SVD of M
           s.t. A'A == 𝕀
       = V U'
    """
    config = config or RAGLiteConfig()
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        # Get random evals from the database.
        evals = session.exec(
            select(Eval).order_by(Eval.id).limit(max(8, max_triplets // max_triplets_per_eval))
        ).all()
        if len(evals) * max_triplets_per_eval < config.embedder.n_embd():
            error_message = "First run `insert_evals()` to generate sufficient Evals."
            raise ValueError(error_message)
        # Loop over the evals to generate (q, p, n) triplets.
        Q = np.zeros((0, config.embedder.n_embd()))  # We want double precision here.  # noqa: N806
        P = np.zeros_like(Q)  # noqa: N806
        N = np.zeros_like(Q)  # noqa: N806
        for eval_ in tqdm(
            evals, desc="Extracting triplets from evals", unit="eval", dynamic_ncols=True
        ):
            # Embed the question.
            question_embedding = embed_strings([eval_.question], config=config)
            # Retrieve chunks that would be used to answer the question.
            chunk_rowids, _ = vector_search(
                eval_.question, num_results=optimize_top_k, query_adapter=False, config=config
            )
            retrieved_chunks = [
                session.exec(select(Chunk).offset(chunk_rowid - 1)).first()
                for chunk_rowid in chunk_rowids
            ]
            # Extract (q, p, n) triplets by comparing the retrieved chunks with the eval.
            num_triplets = 0
            for i, retrieved_chunk in enumerate(retrieved_chunks):
                # Raise an error if the retrieved chunk is None.
                if retrieved_chunk is None:
                    error_message = (
                        f"The chunk with rowid {chunk_rowids[i]} is missing from the database."
                    )
                    raise ValueError(error_message)
                # Select irrelevant chunks.
                if retrieved_chunk.id not in eval_.chunk_ids:
                    # Look up all positive chunks that are ranked lower than this negative one.
                    p_mve = [
                        np.mean(chunk.multi_vector_embedding, axis=0, keepdims=True)
                        for chunk in retrieved_chunks[i + 1 :]
                        if chunk is not None and chunk.id in eval_.chunk_ids
                    ]
                    if not p_mve:
                        continue
                    p = np.vstack(p_mve)
                    n = np.repeat(
                        np.mean(retrieved_chunk.multi_vector_embedding, axis=0, keepdims=True),
                        p.shape[0],
                        axis=0,
                    )
                    q = np.repeat(question_embedding, p.shape[0], axis=0)
                    num_triplets += p.shape[0]
                    # Append the (query, positive, negative) tuples to the Q, P, N matrices.
                    Q = np.vstack([Q, q])  # noqa: N806
                    P = np.vstack([P, p])  # noqa: N806
                    N = np.vstack([N, n])  # noqa: N806
                    # Check if we have sufficient triplets for this eval.
                    if num_triplets >= max_triplets_per_eval:
                        break
            # Check if we have sufficient triplets to compute the query adapter.
            if Q.shape[0] > max_triplets:
                Q, P, N = Q[:max_triplets, :], P[:max_triplets, :], N[:max_triplets, :]  # noqa: N806
                break
        # Normalise the rows of Q, P, N.
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)  # noqa: N806
        P /= np.linalg.norm(P, axis=1, keepdims=True)  # noqa: N806
        N /= np.linalg.norm(N, axis=1, keepdims=True)  # noqa: N806
        # Compute the optimal query adapter A*.
        # TODO: Matmul in float16 is extremely slow compared to single or double precision, why?
        MT = (P - N).T @ Q  # noqa: N806
        if config.vector_search_index_metric == "dot":
            # Use the relaxed Procrustes solution.
            A_star = MT / np.linalg.norm(MT, ord="fro")  # noqa: N806
        elif config.vector_search_index_metric == "cosine":
            # Use the orthogonal Procrustes solution.
            U, _, VT = np.linalg.svd(MT, full_matrices=False)  # noqa: N806
            A_star = U @ VT  # noqa: N806
        else:
            error_message = f"Unsupported ANN metric: {config.vector_search_index_metric}"
            raise ValueError(error_message)
        # Store the optimal query adapter in the database.
        vector_search_chunk_index = session.get(
            VectorSearchChunkIndex, config.vector_search_index_id
        ) or VectorSearchChunkIndex(id=config.vector_search_index_id)
        vector_search_chunk_index.query_adapter = A_star
        session.add(vector_search_chunk_index)
        session.commit()


def answer_evals(
    search: Callable[[str], tuple[list[int], list[float]]] = hybrid_search,
    *,
    config: RAGLiteConfig | None = None,
) -> dict[str, list[str | list[str]]]:
    """Read evals from the database and convert them to a Ragas test set."""
    # Read all evals from the database.
    config = config or RAGLiteConfig()
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        evals = session.exec(select(Eval)).all()
    # Answer evals with RAG.
    answers: list[str] = []
    for eval_ in tqdm(evals, desc="Answering evals", unit="eval", dynamic_ncols=True):
        response = rag(eval_.question, search=search, config=config)
        answer = "".join(response)
        answers.append(answer)
    # Evaluate the answers.
    test_set: dict[str, list[str | list[str]]] = {
        "question": [eval_.question for eval_ in evals],
        "answer": answers,  # type: ignore[dict-item]
        "contexts": [eval_.contexts for eval_ in evals],
        "ground_truth": [eval_.ground_truth for eval_ in evals],
    }
    return test_set
