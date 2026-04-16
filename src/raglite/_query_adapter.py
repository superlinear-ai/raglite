"""Compute and update an optimal query adapter."""

# ruff: noqa: N806

from dataclasses import replace

import numpy as np
from scipy.optimize import lsq_linear
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Eval, IndexMetadata, create_database_engine
from raglite._embed import embed_strings
from raglite._search import vector_search
from raglite._typing import FloatMatrix, FloatVector


def _optimize_query_target(
    q: FloatVector,
    P: FloatMatrix,  # noqa: N803,
    N: FloatMatrix,  # noqa: N803,
    *,
    α: float = 0.05,  # noqa: PLC2401
) -> FloatVector:
    # Convert to double precision for the optimizer.
    q_dtype = q.dtype
    q, P, N = q.astype(np.float64), P.astype(np.float64), N.astype(np.float64)
    # Construct the constraint matrix D := P - (1 + α) * N.  # noqa: RUF003
    D = np.reshape(P[:, np.newaxis, :] - (1.0 + α) * N[np.newaxis, :, :], (-1, P.shape[1]))
    # Solve the dual problem min_μ ½ ‖q + Dᵀ μ‖² s.t. μ ≥ 0.
    A, b = D.T, -q
    μ_star = lsq_linear(A, b, bounds=(0.0, np.inf), tol=np.finfo(A.dtype).eps).x  # noqa: PLC2401
    # Recover the primal solution q* = q + Dᵀ μ*.
    q_star: FloatVector = (q + D.T @ μ_star).astype(q_dtype)
    return q_star


def update_query_adapter(
    *,
    max_evals: int = 4096,
    optimize_top_k: int = 40,
    optimize_gap: float = 0.05,
    config: RAGLiteConfig | None = None,
) -> FloatMatrix:
    """Compute an optimal query adapter and update the database with it.

    This function computes an optimal linear transform A, called a 'query adapter', that is used to
    transform a query embedding q as A @ q before searching for the nearest neighbouring chunks in
    order to improve the quality of the search results.

    Given a set of triplets (qᵢ, pᵢ, nᵢ), we want to find the query adapter A that increases the
    score pᵢᵀqᵢ of the positive chunk pᵢ and decreases the score nᵢᵀqᵢ of the negative chunk nᵢ.

    If the nearest neighbour search uses the dot product as its relevance score, we can find the
    optimal query adapter by solving the following relaxed Procrustes optimisation problem with a
    bound on the Frobenius norm of A:

    A* := argmax Σᵢ pᵢᵀ (A qᵢ) - nᵢᵀ (A qᵢ)
                 Σᵢ (pᵢ - nᵢ)ᵀ A qᵢ
                 trace[ (P - N) A Qᵀ ]  where  Q := [q₁ᵀ; ...; qₖᵀ]
                                               P := [p₁ᵀ; ...; pₖᵀ]
                                               N := [n₁ᵀ; ...; nₖᵀ]
                 trace[ Qᵀ (P - N) A ]
                 trace[ Mᵀ A ]          where  M := (P - N)ᵀ Q
            s.t. ||A||_F == 1
        = M / ||M||_F

    If the nearest neighbour search uses the cosine similarity as its relevance score, we can find
    the optimal query adapter by solving the following orthogonal Procrustes optimisation problem
    [1] with an orthogonality constraint on A:

    A* := argmax Σᵢ pᵢᵀ (A qᵢ) - nᵢᵀ (A qᵢ)
                 Σᵢ (pᵢ - nᵢ)ᵀ A qᵢ
                 trace[ (P - N) A Qᵀ ]
                 trace[ Qᵀ (P - N) A ]
                 trace[ Mᵀ A ]
                 trace[ (U Σ V)ᵀ A ]      where  U Σ Vᵀ := M is the SVD of M
                 trace[ Σ V A Uᵀ ]
            s.t. AᵀA == 𝕀
        = U Vᵀ

    The action of A* is to map a query embedding qᵢ to a target vector t := (pᵢ - nᵢ) that maximally
    separates the positive and negative chunks. For a given query embedding qᵢ, a retrieval method
    will yield a result set containing both positive and negative chunks. Instead of extracting
    multiple triplets (qᵢ, pᵢ, nᵢ) from each such result set, we can compute a single optimal target
    vector t* for the query embedding qᵢ as follows:

    t* := argmax ½ ||t - qᵢ||²
            s.t. Dᵢ t >= 0

    where the constraint matrix Dᵢ := [pₘᵀ - (1 + α) nₙᵀ]ₘₙ comprises all pairs of positive and
    negative chunk embeddings in the result set corresponding to the query embedding qᵢ. This
    optimisation problem expresses the idea that the target vector t* should be as close as
    possible to the query embedding qᵢ, while separating all positive and negative chunk embeddings
    in the result set by a margin of at least α. To solve this problem, we'll first introduce
    a Lagrangian with Lagrange multipliers μ:

    L(t, μ) := ½ ||t - qᵢ||² + μᵀ (-Dᵢ t)

    Now we can set the gradient of the Lagrangian to zero to find the optimal target vector t*:

    ∇ₜL = t - qᵢ - Dᵢᵀ μ = 0
    t* = qᵢ + Dᵢᵀ μ*

    where μ* is the solution to the dual nonnegative least squares problem

    μ* := argmin ½ ||qᵢ + Dᵢᵀ μ||²
            s.t. μ >= 0

    Parameters
    ----------
    max_evals
        The maximum number of evals to use to compute the query adapter. Each eval corresponds to a
        rank-one update of the query adapter A.
    optimize_top_k
        The number of search results per eval to optimize.
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
    """
    config = config or RAGLiteConfig()
    config_no_query_adapter = replace(config, vector_search_query_adapter=False)
    with Session(engine := create_database_engine(config)) as session:
        # Get random evals from the database.
        chunk_embedding = session.exec(select(ChunkEmbedding).limit(1)).first()
        if chunk_embedding is None:
            error_message = "First run `insert_documents()` to insert documents."
            raise ValueError(error_message)
        evals = session.exec(select(Eval).order_by(Eval.id).limit(max_evals)).all()
        if len(evals) == 0:
            error_message = "First run `insert_evals()` to generate evals."
            raise ValueError(error_message)
        # Construct the query and target matrices.
        Q = np.zeros((0, len(chunk_embedding.embedding)))
        T = np.zeros_like(Q)
        for eval_ in tqdm(
            evals, desc="Optimizing evals", unit="eval", dynamic_ncols=True, leave=False
        ):
            # Embed the question.
            q = embed_strings([eval_.question], config=config)[0]
            # Retrieve chunks that would be used to answer the question.
            chunk_ids, _ = vector_search(
                q, num_results=optimize_top_k, config=config_no_query_adapter
            )
            retrieved_chunks = session.exec(select(Chunk).where(col(Chunk.id).in_(chunk_ids))).all()
            retrieved_chunks = sorted(retrieved_chunks, key=lambda chunk: chunk_ids.index(chunk.id))
            # Skip this eval if it doesn't contain both relevant and irrelevant chunks.
            is_relevant = np.array([chunk.id in eval_.chunk_ids for chunk in retrieved_chunks])
            if not np.any(is_relevant) or not np.any(~is_relevant):
                continue
            # Extract the positive and negative chunk embeddings.
            P = np.vstack(
                [
                    chunk.embedding_matrix[[np.argmax(chunk.embedding_matrix @ q)]]
                    for chunk in np.array(retrieved_chunks)[is_relevant]
                ]
            )
            N = np.vstack(
                [
                    chunk.embedding_matrix[[np.argmax(chunk.embedding_matrix @ q)]]
                    for chunk in np.array(retrieved_chunks)[~is_relevant]
                ]
            )
            # Compute the optimal target vector t for this query embedding q.
            t = _optimize_query_target(q, P, N, α=optimize_gap)
            Q = np.vstack([Q, q[np.newaxis, :]])
            T = np.vstack([T, t[np.newaxis, :]])
        # Normalise the rows of Q and T.
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        if config.vector_search_distance_metric == "cosine":
            T /= np.linalg.norm(T, axis=1, keepdims=True)
        # Compute the optimal unconstrained query adapter M.
        n, d = Q.shape
        M = (1 / n) * T.T @ Q
        if n < d or np.linalg.matrix_rank(Q) < d:
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
        # Clear the index metadata cache to allow the new query adapter to be used.
        IndexMetadata._get.cache_clear()  # noqa: SLF001
    return A_star
