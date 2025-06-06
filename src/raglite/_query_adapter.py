"""Compute and update an optimal query adapter."""

# ruff: noqa: N803, N806, PLC2401, PLR0913, RUF003

import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import partial

import numpy as np
from scipy.optimize import lsq_linear, minimize
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Eval, IndexMetadata, create_database_engine
from raglite._embed import embed_strings
from raglite._search import vector_search
from raglite._typing import FloatMatrix, FloatTensor, FloatVector


def _extract_triplets(
    eval_: Eval, *, optimize_top_k: int, config: RAGLiteConfig
) -> tuple[FloatVector, FloatMatrix, FloatMatrix]:
    with Session(create_database_engine(config)) as session:
        # Embed the question.
        q = embed_strings([eval_.question], config=config)[0]
        # Retrieve chunks that would be used to answer the question.
        chunk_ids, _ = vector_search(q, num_results=optimize_top_k, config=config)
        retrieved_chunks = session.exec(select(Chunk).where(col(Chunk.id).in_(chunk_ids))).all()
        retrieved_chunks = sorted(retrieved_chunks, key=lambda chunk: chunk_ids.index(chunk.id))
        # Skip this eval if it doesn't contain both relevant and irrelevant chunks.
        is_relevant = np.array([chunk.id in eval_.chunk_ids for chunk in retrieved_chunks])
        if not np.any(is_relevant) or not np.any(~is_relevant):
            error_message = "Eval does not contain both relevant and irrelevant chunks."
            raise ValueError(error_message)
        # Skip this eval if it is perfectly ranked.
        num_relevant = np.sum(is_relevant)
        if np.sum(is_relevant[:num_relevant]) == num_relevant:
            error_message = "Eval is already perfectly ranked."
            raise ValueError(error_message)
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
    return q, P, N


def _optimize_query_target(
    q: FloatVector,
    P: FloatMatrix,
    N: FloatMatrix,
    *,
    α: float = 0.05,
) -> FloatVector:
    # Convert to double precision for the optimizer.
    q, P, N = q.astype(np.float64), P.astype(np.float64), N.astype(np.float64)
    # Construct the constraint matrix D := P - (1 + α) * N.
    D = np.reshape(P[:, np.newaxis, :] - (1.0 + α) * N[np.newaxis, :, :], (-1, P.shape[1]))
    # Solve the dual problem min_μ ½ ‖q + Dᵀ μ‖² s.t. μ ≥ 0.
    A, b = D.T, -q
    result = lsq_linear(A, b, bounds=(0.0, np.inf), tol=np.finfo(A.dtype).eps)
    μ_star = result.x
    # Recover the primal solution q* = q + Dᵀ μ*.
    q_star: FloatVector = q + D.T @ μ_star
    return q_star


def _compute_query_adapter(
    w: FloatVector, Q: FloatMatrix, T: FloatMatrix, PT: FloatMatrix, config: RAGLiteConfig
) -> FloatMatrix:
    # Compute the optimal unconstrained query adapter M.
    n, d = Q.shape
    M = (1.0 / n) * T.T @ (w[:, np.newaxis] * Q) + PT
    # Compute the optimal constrained query adapter A* from M, given the distance metric.
    A_star: FloatMatrix
    if config.vector_search_distance_metric == "dot":
        # Use the relaxed Procrustes solution.
        A_star = M / np.linalg.norm(M, ord="fro") * np.sqrt(d)
    elif config.vector_search_distance_metric == "cosine":
        # Use the orthogonal Procrustes solution.
        U, _, VT = np.linalg.svd(M, full_matrices=False)
        A_star = U @ VT
    return A_star


def _compute_query_adapter_grad(
    w: FloatVector,
    Q: FloatMatrix,
    T: FloatMatrix,
    PT: FloatMatrix,
    config: RAGLiteConfig,
) -> FloatTensor:
    n, d = Q.shape
    M = (1.0 / n) * T.T @ (w[:, np.newaxis] * Q) + PT
    dAdw: FloatTensor  # n × d × d
    if config.vector_search_distance_metric == "dot":
        fro = np.linalg.norm(M, ord="fro")
        if fro <= np.sqrt(np.finfo(M.dtype).eps):
            return np.zeros((n, d, d), dtype=M.dtype)
        outer = (T[:, :, np.newaxis] * Q[:, np.newaxis, :]) / n
        inner = np.sum(outer * M[np.newaxis, :, :], axis=(1, 2))
        dAdw = (outer - (inner / fro**2)[:, np.newaxis, np.newaxis] * M) / fro * np.sqrt(d)
    elif config.vector_search_distance_metric == "cosine":
        U, σ, VT = np.linalg.svd(M, full_matrices=False)
        P = T @ U
        R = Q @ VT.T
        skew = P[:, :, np.newaxis] * R[:, np.newaxis, :] - R[:, :, np.newaxis] * P[:, np.newaxis, :]
        denom = σ[:, np.newaxis] + σ[np.newaxis, :]
        denom[denom <= (np.finfo(M.dtype).eps ** 0.25) * σ[0]] = np.inf
        core = skew / denom[np.newaxis, :, :] / n
        dAdw = np.einsum("ab,nbc,cd->nad", U, core, VT, optimize=True)
    return dAdw


def _objective_function(
    w: FloatVector,
    Q_train: FloatMatrix,
    T_train: FloatMatrix,
    PT_train: FloatMatrix,
    Q_val: FloatMatrix,
    D_val: FloatMatrix,
    config: RAGLiteConfig,
) -> float:
    A = _compute_query_adapter(w, Q_train, T_train, PT_train, config)
    gap: float = np.mean(np.sum(D_val * ((Q_val @ A.T) - Q_val), axis=1))
    return -gap


def _gradient(
    w: FloatVector,
    Q_train: FloatMatrix,
    T_train: FloatMatrix,
    PT_train: FloatMatrix,
    Q_val: FloatMatrix,
    D_val: FloatMatrix,
    config: RAGLiteConfig,
) -> FloatVector:
    dAdw = _compute_query_adapter_grad(w, Q_train, T_train, PT_train, config)  # n × d × d
    grad: FloatVector = -np.einsum("nij,mj,mi->n", dAdw, Q_val, D_val, optimize=True) / len(Q_val)
    return grad


def update_query_adapter(  # noqa: PLR0915
    *,
    max_evals: int = 4096,
    optimize_top_k: int = 40,
    gap_margin: float = 0.05,
    gap_max_iter: int = 20,
    gap_tol: float = 1e-6,
    gap_validation_size: float = 0.25,
    max_workers: int | None = None,
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

    Finally, we weight the optimal target vectors with a set of weights w* that are optimised to
    maximise the gap between the positive and negative chunks in a validation set of evals:

    w* := argmax ΣᵢΣₘₙ (pₘ⁽ᵛᵃˡ⁾ - nₙ⁽ᵛᵃˡ⁾)ᵀ A(w) qᵢ⁽ᵛᵃˡ⁾

    where A(w) is the weighted query adapter (1 / n) Tᵀ diag(w) Q + P, T is the matrix of optimal
    target vectors t*, and P is the passthrough matrix 𝕀 - Qᵀ (Q Qᵀ)⁺ Q that lets query vectors
    outside of the row space of Q through unaffected.

    Parameters
    ----------
    max_evals
        The maximum number of evals to use to compute the query adapter. Each eval corresponds to a
        rank-one update of the query adapter A.
    optimize_top_k
        The number of search results per eval to optimize.
    gap_margin
        The margin α to use when computing the optimal query target t* for each query embedding qᵢ.
    gap_max_iter
        The maximum number of iterations to use to optimize the query target weights w*.
    gap_tol
        The tolerance to use when optimizing the query target weights w*.
    gap_validation_size
        The fraction of evals to use for optimizing the query target weights w*. The remaining evals
        are used for training the unweighted query adapter.
    max_workers
        The maximum number of worker threads to use for triplet extraction and query target
        optimization.
    config
        The RAGLite config to use to construct and store the query adapter.

    Raises
    ------
    ValueError
        If no documents have been inserted into the database yet.
    ValueError
        If no evals have been inserted into the database yet.
    ValueError
        If no evals are found with incorrectly ranked results to optimize.
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
        # Collect triplets (qᵢ, Pᵢ × Nᵢ) for each eval.
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(
                total=len(evals), desc="Extracting triplets", unit="eval", dynamic_ncols=True
            ) as pbar,
        ):
            q: list[FloatVector] = []
            P: list[FloatMatrix] = []
            N: list[FloatMatrix] = []
            futures = [
                executor.submit(
                    partial(
                        _extract_triplets,
                        optimize_top_k=optimize_top_k,
                        config=config_no_query_adapter,
                    ),
                    eval_,
                )
                for eval_ in evals
            ]
            for future in as_completed(futures):
                with contextlib.suppress(Exception):
                    pbar.update()
                    qi, Pi, Ni = future.result()
                    q.append(qi)
                    P.append(Pi)
                    N.append(Ni)
        # Exit if there are no triplets to optimise.
        if len(q) == 0:
            error_message = "No evals found with incorrectly ranked results to optimize."
            raise ValueError(error_message)
        # Split in train and validation sets.
        val_size = round(gap_validation_size * len(q))
        train_size = len(q) - val_size
        # Compute the optimal query targets T.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            T_train = np.vstack(
                list(
                    tqdm(
                        executor.map(
                            partial(_optimize_query_target, α=gap_margin),
                            q[:train_size],
                            P[:train_size],
                            N[:train_size],
                        ),
                        total=train_size,
                        desc="Optimizing query targets",
                        unit="query",
                        dynamic_ncols=True,
                    )
                )
            )
        # Normalise the rows of Q and T.
        Q = np.vstack(q).astype(np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        if config.vector_search_distance_metric == "cosine":
            T_train /= np.linalg.norm(T_train, axis=1, keepdims=True)
        # Search for the optimal gap α* on a subset of the triplets.
        w_star = np.ones(train_size)
        if val_size >= 1:
            # Split the triplets in a train and validation set.
            Q_train = Q[:train_size]
            Q_val, P_val, N_val = Q[-val_size:], P[-val_size:], N[-val_size:]
            # Compute a passthrough matrix.
            n, d = Q_train.shape
            if n < d or np.linalg.matrix_rank(Q) < d:
                PT_train = np.eye(d) - Q_train.T @ np.linalg.pinv(Q_train @ Q_train.T) @ Q_train
            else:
                PT_train = np.zeros((d, d))
            # Construct the delta matrix D[i, :] := mean([pₘᵀ - nₙᵀ]ₘₙ, axis=0).
            D_val = np.empty_like(Q_val)
            for i, (Pi, Ni) in enumerate(zip(P_val, N_val, strict=True)):
                D_val[i] = np.mean(np.reshape(Pi[:, None, :] - Ni[None, :, :], (-1, d)), axis=0)
            # Compute the optimal gap α*.
            with tqdm(
                total=gap_max_iter,
                desc="Optimizing query target weights",
                unit="iter",
                dynamic_ncols=True,
            ) as pbar:
                result = minimize(
                    _objective_function,
                    jac=_gradient,
                    x0=np.ones(train_size),
                    args=(Q_train, T_train, PT_train, Q_val, D_val, config_no_query_adapter),
                    method="L-BFGS-B",
                    callback=lambda intermediate_result: (
                        pbar.update(),
                        pbar.set_postfix({"gap": -intermediate_result.fun}),
                    ),
                    options={"ftol": gap_tol, "maxiter": gap_max_iter, "maxcor": 10, "maxls": 10},
                )
            w_star = result.x
        # Compute the optimal query adapter.
        A_star = _compute_query_adapter(w_star, Q_train, T_train, PT_train, config)
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
