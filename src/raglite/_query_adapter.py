"""Compute and update an optimal query adapter."""

# ruff: noqa: N806

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Eval, IndexMetadata, create_database_engine
from raglite._embed import embed_strings
from raglite._search import vector_search
from raglite._typing import FloatMatrix


def update_query_adapter(  # noqa: C901, PLR0912, PLR0915
    *,
    max_triplets: int = 4096,
    max_triplets_per_eval: int = 64,
    optimize_top_k: int = 40,
    optimize_gap: float = 0.6,
    config: RAGLiteConfig | None = None,
) -> FloatMatrix:
    """Compute an optimal query adapter and update the database with it.

    This function computes an optimal linear transform A, called a 'query adapter', that is used to
    transform a query embedding q as A @ q before searching for the nearest neighbouring chunks in
    order to improve the quality of the search results.

    Given a set of triplets (qᵢ, pᵢ, nᵢ), we want to find the query adapter A that increases the
    score pᵢ'qᵢ of the positive chunk pᵢ and decreases the score nᵢ'qᵢ of the negative chunk nᵢ.

    If the nearest neighbour search uses the dot product as its relevance score, we can find the
    optimal query adapter by solving the following relaxed Procrustes optimisation problem [1] with
    a bound on the Frobenius norm of A:

    A* := argmax Σᵢ pᵢ' (A qᵢ) - nᵢ' (A qᵢ)
                 Σᵢ (pᵢ - nᵢ)' A qᵢ
                 trace[ (P - N) A Q' ]  where  Q := [q₁'; ...; qₖ']
                                               P := [p₁'; ...; pₖ']
                                               N := [n₁'; ...; nₖ']
                 trace[ Q' (P - N) A ]
                 trace[ M' A ]          where  M := (P - N)' Q
            s.t. ||A||_F == 1
        = M / ||M||_F

    If the nearest neighbour search uses the cosine similarity as its relevance score, we can find
    the optimal query adapter by solving the following orthogonal Procrustes optimisation problem
    [1] with an orthogonality constraint on A:

    A* := argmax Σᵢ pᵢ' (A qᵢ) - nᵢ' (A qᵢ)
                 Σᵢ (pᵢ - nᵢ)' A qᵢ
                 trace[ (P - N) A Q' ]
                 trace[ Q' (P - N) A ]
                 trace[ M' A ]
                 trace[ (U Σ V)' A ]      where  U Σ V' := M is the SVD of M
                 trace[ Σ V A U' ]
            s.t. A'A == 𝕀
        = U V'

    The action of A* is to map a query embedding qᵢ to a target vector (pᵢ - nᵢ) that maximally
    separates the positive and negative chunks. An additional requirement on A* is that we want to
    limit its effect so that it adjusts q just enough to invert incorrectly ordered (q, p, n)
    triplets, but not so much as to affect the correctly ordered ones. To achieve this, we'll
    rewrite the target vector sᵢ(t) as a slerp [2] from qᵢ to (pᵢ - nᵢ) with interpolation parameter
    t ∈ [0, 1]:

    θᵢ := arccos((pᵢ - nᵢ)' qᵢ / ||pᵢ - nᵢ|| ||qᵢ||)
    lᵢ(t) := sin((1 - t) θᵢ) / sin(θᵢ)
    rᵢ(t) := sin(t θᵢ) / sin(θᵢ)
    sᵢ(t) := lᵢ(t) qᵢ + rᵢ(t) (pᵢ - nᵢ)

    We want to choose the smallest tᵢ such that pᵢ' sᵢ(tᵢ) > (1 + α) n' sᵢ(tᵢ) for some choice of
    threshold α and reference negative chunk embedding n:

    pᵢ' sᵢ(tᵢ) > (1 + α) n' sᵢ(tᵢ)
    [sin((1 - tᵢ) θᵢ) / sin(θᵢ)] (pᵢ - (1 + α) n)' qᵢ +
    [sin(tᵢ θᵢ)       / sin(θᵢ)] (pᵢ - (1 + α) n)' (pᵢ - nᵢ) > 0

    Let S := sin(θᵢ), C := cos(θᵢ), X := (pᵢ - n)' qᵢ, and Y := (pᵢ - n)' (pᵢ - nᵢ), then:

    [S cos(tᵢ θᵢ) - C sin(tᵢ θᵢ)] X / S + sin(tᵢ θᵢ) Y / S > 0
    tᵢ = min_{k ∈ ℤ} [tan⁻¹(SX / (CX - Y)) + kπ] / θᵢ s.t. t ∈ [0, 1]

    We can then redefine the unconstrained query adapter matrix M as:

    M := [k⁻¹ diag(lᵢ(tᵢ)) Q + k⁻¹ diag(rᵢ(tᵢ)) (P - N)]' Q + E
    E := 𝕀 - Q' (Q Q')⁺ Q

    where Q is row-normalised before applying the slerp, k⁻¹ computes the mean contribution of the
    k triplets, and E is an additional passthrough term that maps the query embedding qᵢ to itself
    if it is not in the row space of Q. In other words, when the query is dissimilar from the evals,
    the query adapter passes the query embedding through unchanged.

    Parameters
    ----------
    max_triplets
        The maximum number of (q, p, n) triplets to compute. Each triplet corresponds to a rank-one
        update of the query adapter A.
    max_triplets_per_eval
        The maximum number of (q, p, n) triplets a single eval may contribute to the query adapter.
    optimize_top_k
        The number of search results per eval to extract (q, p, n) triplets from.
    optimize_gap
        The strength of the query adapter, expressed as a nonnegative number. Should be large enough
        to correct incorrectly ranked results, but small enough to not affect correctly ranked
        results.
    config
        The RAGLite config to use to construct and store the query adapter.

    Raises
    ------
    ValueError
        If no documents have been inserted into the database yet.
    ValueError
        If no evals have been inserted into the database yet.
    ValueError
        If the `config.vector_search_distance_metric` is not supported.

    Returns
    -------
    FloatMatrix
        The query adapter.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    [2] https://en.wikipedia.org/wiki/Slerp
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
        evals = session.exec(select(Eval).order_by(Eval.id).limit(max_triplets)).all()
        if len(evals) == 0:
            error_message = "First run `insert_evals()` to generate evals."
            raise ValueError(error_message)
        # Loop over the evals to generate (q, p, n) triplets.
        Q = np.zeros((0, len(chunk_embedding.embedding)))
        P = np.zeros_like(Q)
        N = np.zeros_like(Q)
        F = np.zeros_like(Q)
        for eval_ in tqdm(
            evals, desc="Extracting triplets from evals", unit="eval", dynamic_ncols=True
        ):
            # Embed the question.
            question_embedding = embed_strings([eval_.question], config=config)
            # Retrieve chunks that would be used to answer the question.
            chunk_ids, _ = vector_search(
                question_embedding[0], num_results=optimize_top_k, config=config_no_query_adapter
            )
            retrieved_chunks = session.exec(select(Chunk).where(col(Chunk.id).in_(chunk_ids))).all()
            retrieved_chunks = sorted(retrieved_chunks, key=lambda chunk: chunk_ids.index(chunk.id))
            # Extract (q, p, n) triplets from the eval.
            n_first = None
            num_triplets = 0
            for i, retrieved_chunk in enumerate(retrieved_chunks):
                # Only loop over irrelevant chunks.
                if retrieved_chunk.id not in eval_.chunk_ids:
                    continue
                irrelevant_chunk = retrieved_chunk
                # Grab the negative chunk embedding of this irrelevant chunk.
                n_top = irrelevant_chunk.embedding_matrix[
                    [np.argmax(irrelevant_chunk.embedding_matrix @ question_embedding.T)]
                ]
                if n_first is None:
                    n_first = n_top
                # Grab the positive chunk embeddings that are ranked lower than the negative one.
                p_top = [
                    chunk.embedding_matrix[
                        [np.argmax(chunk.embedding_matrix @ question_embedding.T)]
                    ]
                    for chunk in retrieved_chunks[i + 1 :]  # Chunks that are ranked lower.
                    if chunk is not None and chunk.id in eval_.chunk_ids
                ]
                # Ensure that we only have (q, p, n) triplets for which p is ranked lower than n.
                p_top = [p for p in p_top if (n_top - p) @ question_embedding.T > 0]
                if not p_top:
                    continue
                # Stack the (q, p, n) triplets.
                p = np.vstack(p_top)
                f = np.repeat(n_first, p.shape[0], axis=0)
                n = np.repeat(n_top, p.shape[0], axis=0)
                q = np.repeat(question_embedding, p.shape[0], axis=0)
                num_triplets += p.shape[0]
                # Append the (q, p, n) triplets to the Q, P, N matrices.
                Q = np.vstack([Q, q])
                P = np.vstack([P, p])
                N = np.vstack([N, n])
                F = np.vstack([F, f])
                # Stop if we have enough triplets for this eval.
                if num_triplets >= max_triplets_per_eval:
                    break
            # Stop if we have enough triplets in total.
            if Q.shape[0] > max_triplets:
                Q = Q[:max_triplets, :]
                P = P[:max_triplets, :]
                N = N[:max_triplets, :]
                F = F[:max_triplets, :]
                break
        # Normalise the rows of Q.
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        # Compute the optimal slerp interpolation parameter tᵢ for each triplet.
        θ = np.arccos(np.sum(((P - N) / np.linalg.norm(P - N, axis=1, keepdims=True) * Q), axis=1))  # noqa: PLC2401
        X = np.sum((P - (1 + optimize_gap) * F) * Q, axis=1)
        Y = np.sum((P - (1 + optimize_gap) * F) * (P - N), axis=1)
        t = np.arctan2(np.sin(θ) * X, np.cos(θ) * X - Y)  # t ∈ [-π, π]
        t[t < 0] += np.pi  # t ∈ [0, π]
        t[θ > 0] /= θ[θ > 0]  # θ ∈ [0, π]
        t = np.clip(t, 0, 1)
        # Compute the slerp coefficients lᵢ(tᵢ) and rᵢ(tᵢ).
        l = np.sin((1 - t) * θ) / np.sin(θ)  # noqa: E741
        r = np.sin(t * θ) / np.sin(θ)
        # Compute the optimal unconstrained query adapter M.
        k, d = Q.shape
        M = (1 / k) * (l[:, np.newaxis] * Q + r[:, np.newaxis] * (P - N)).T @ Q
        if len(evals) < d or np.linalg.matrix_rank(Q) < d:
            M += np.eye(d) - Q.T @ np.linalg.pinv(Q @ Q.T) @ Q
        # Compute the optimal constrained query adapter A* from M, given the distance metric.
        A_star: FloatMatrix
        if config.vector_search_distance_metric == "dot":
            # Use the relaxed Procrustes solution.
            A_star = M / np.linalg.norm(M, ord="fro") * np.sqrt(d)
        elif config.vector_search_distance_metric == "cosine":
            # Use the orthogonal Procrustes solution.
            U, _, VT = np.linalg.svd(M, full_matrices=False)
            A_star = U @ VT
        else:
            error_message = f"Unsupported metric: {config.vector_search_distance_metric}"
            raise ValueError(error_message)
        # Store the optimal query adapter in the database.
        index_metadata = session.get(IndexMetadata, "default") or IndexMetadata(id="default")
        index_metadata.metadata_["query_adapter"] = A_star
        flag_modified(index_metadata, "metadata_")
        session.add(index_metadata)
        session.commit()
        if engine.dialect.name == "duckdb":
            session.execute(text("CHECKPOINT;"))
    return A_star
