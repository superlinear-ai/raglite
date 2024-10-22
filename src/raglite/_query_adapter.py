"""Compute and update an optimal query adapter."""

import numpy as np
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Eval, IndexMetadata, create_database_engine
from raglite._embed import embed_sentences
from raglite._search import vector_search


def update_query_adapter(  # noqa: PLR0915, C901
    *,
    max_triplets: int = 4096,
    max_triplets_per_eval: int = 64,
    optimize_top_k: int = 40,
    config: RAGLiteConfig | None = None,
) -> None:
    """Compute an optimal query adapter and update the database with it.

    This function computes an optimal linear transform A, called a 'query adapter', that is used to
    transform a query embedding q as A @ q before searching for the nearest neighbouring chunks in
    order to improve the quality of the search results.

    Given a set of triplets (qᵢ, pᵢ, nᵢ), we want to find the query adapter A that increases the
    score pᵢ'qᵢ of the positive chunk pᵢ and decreases the score nᵢ'qᵢ of the negative chunk nᵢ.

    If the nearest neighbour search uses the dot product as its relevance score, we can find the
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

    If the nearest neighbour search uses the cosine similarity as its relevance score, we can find
    the optimal query adapter by solving the following orthogonal Procrustes optimisation problem
    with an orthogonality constraint on A:

    A* = argmax Σᵢ pᵢ' (A qᵢ) - nᵢ' (A qᵢ)
                Σᵢ (pᵢ - nᵢ)' A qᵢ
                trace[ (P - N) A Q' ]
                trace[ Q' (P - N) A ]
                trace[ M A ]
                trace[ U Σ V' A ]      where  U Σ V' := M is the SVD of M
                trace[ Σ V' A U ]
           s.t. A'A == 𝕀
       = V U'

    Additionally, we want to limit the effect of A* so that it adjusts q just enough to invert
    incorrectly ordered (q, p, n) triplets, but not so much as to affect the correctly ordered ones.
    To achieve this, we'll rewrite M as α(M / s) + (1 - α)𝕀, where s scales M to the same norm as 𝕀,
    and choose the smallest α that ranks (q, p, n) correctly. If α = 0, the relevance score gap
    between an incorrect (p, n) pair would be B := (p - n)' q < 0. If α = 1, the relevance score gap
    would be A := (p - n)' (p - n) / ||p - n|| > 0. For a target relevance score gap of say
    C := 5% * A, the optimal α is then given by αA + (1 - α)B = C => α = (B - C) / (B - A).
    """
    config = config or RAGLiteConfig()
    config_no_query_adapter = RAGLiteConfig(
        **{**config.__dict__, "vector_search_query_adapter": False}
    )
    engine = create_database_engine(config)
    with Session(engine) as session:
        # Get random evals from the database.
        chunk_embedding = session.exec(select(ChunkEmbedding).limit(1)).first()
        if chunk_embedding is None:
            error_message = "First run `insert_document()` to insert documents."
            raise ValueError(error_message)
        evals = session.exec(
            select(Eval).order_by(Eval.id).limit(max(8, max_triplets // max_triplets_per_eval))
        ).all()
        if len(evals) * max_triplets_per_eval < len(chunk_embedding.embedding):
            error_message = "First run `insert_evals()` to generate sufficient evals."
            raise ValueError(error_message)
        # Loop over the evals to generate (q, p, n) triplets.
        Q = np.zeros((0, len(chunk_embedding.embedding)))  # noqa: N806
        P = np.zeros_like(Q)  # noqa: N806
        N = np.zeros_like(Q)  # noqa: N806
        for eval_ in tqdm(
            evals, desc="Extracting triplets from evals", unit="eval", dynamic_ncols=True
        ):
            # Embed the question.
            question_embedding = embed_sentences([eval_.question], config=config)
            # Retrieve chunks that would be used to answer the question.
            chunk_ids, _ = vector_search(
                question_embedding, num_results=optimize_top_k, config=config_no_query_adapter
            )
            retrieved_chunks = session.exec(select(Chunk).where(col(Chunk.id).in_(chunk_ids))).all()
            # Extract (q, p, n) triplets by comparing the retrieved chunks with the eval.
            num_triplets = 0
            for i, retrieved_chunk in enumerate(retrieved_chunks):
                # Select irrelevant chunks.
                if retrieved_chunk.id not in eval_.chunk_ids:
                    # Look up all positive chunks (each represented by the mean of its multi-vector
                    # embedding) that are ranked lower than this negative one (represented by the
                    # embedding in the multi-vector embedding that best matches the query).
                    p_mean = [
                        np.mean(chunk.embedding_matrix, axis=0, keepdims=True)
                        for chunk in retrieved_chunks[i + 1 :]
                        if chunk is not None and chunk.id in eval_.chunk_ids
                    ]
                    n_top = retrieved_chunk.embedding_matrix[
                        np.argmax(retrieved_chunk.embedding_matrix @ question_embedding.T),
                        np.newaxis,
                        :,
                    ]
                    # Filter out any (p, n, q) triplets for which the mean positive embedding ranks
                    # higher than the top negative one.
                    p_mean = [p_e for p_e in p_mean if (n_top - p_e) @ question_embedding.T > 0]
                    if not p_mean:
                        continue
                    # Stack the (p, n, q) triplets.
                    p = np.vstack(p_mean)
                    n = np.repeat(n_top, p.shape[0], axis=0)
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
        # Compute the optimal weighted query adapter A*.
        # TODO: Matmul in float16 is extremely slow compared to single or double precision, why?
        gap_before = np.sum((P - N) * Q, axis=1)
        gap_after = 2 * (1 - np.sum(P * N, axis=1)) / np.linalg.norm(P - N, axis=1)
        gap_target = 0.05 * gap_after
        α = (gap_before - gap_target) / (gap_before - gap_after)  # noqa: PLC2401
        MT = (α[:, np.newaxis] * (P - N)).T @ Q  # noqa: N806
        s = np.linalg.norm(MT, ord="fro") / np.sqrt(MT.shape[0])
        MT = np.mean(α) * (MT / s) + np.mean(1 - α) * np.eye(Q.shape[1])  # noqa: N806
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
        index_metadata = session.get(IndexMetadata, "default") or IndexMetadata(id="default")
        index_metadata.metadata_["query_adapter"] = A_star
        flag_modified(index_metadata, "metadata_")
        session.add(index_metadata)
        session.commit()
