"""Split a document into semantic chunks."""

import re

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from raglite._typing import FloatMatrix


def split_chunks(  # noqa: C901, PLR0915
    chunklets: list[str],
    chunklet_embeddings: FloatMatrix,
    max_size: int = 2048,
) -> tuple[list[str], list[FloatMatrix]]:
    """Split chunklets into optimal semantic chunks with corresponding chunklet embeddings.

    A chunk is a concatenated contiguous list of chunklets from a document. This function
    optimally partitions a document into chunks using binary integer programming.

    A partioning of a document into chunks is considered optimal if the total cost of partitioning
    the document into chunks is minimized. The cost of adding a partition point is given by the
    cosine similarity of the chunklet embedding before and after the partition point, corrected by
    the discourse vector of the chunklet embeddings across the document.

    Parameters
    ----------
    chunklets
        The input document as a list of chunklets.
    chunklet_embeddings
        A NumPy array wherein the i'th row is an embedding vector corresponding to the i'th
        chunklet. Embedding vectors are expected to have nonzero length.
    max_size
        The maximum size of a chunk in characters.

    Returns
    -------
    tuple[list[str], list[FloatMatrix]]
        The document and chunklet embeddings optimally partitioned into chunks.
    """
    # Validate the input.
    chunklet_size = np.asarray([len(chunklet) for chunklet in chunklets])
    if not np.all(chunklet_size <= max_size):
        error_message = "Chunklet larger than chunk max_size detected."
        raise ValueError(error_message)
    if not np.all(np.linalg.norm(chunklet_embeddings, axis=1) > 0.0):
        error_message = "Chunklet embeddings with zero norm detected."
        raise ValueError(error_message)
    # Exit early if there is only one chunk to return.
    if len(chunklets) <= 1 or sum(chunklet_size) <= max_size:
        return ["".join(chunklets)] if chunklets else chunklets, [chunklet_embeddings]
    # Normalise the chunklet embeddings to unit norm.
    X = chunklet_embeddings.astype(np.float32)  # noqa: N806
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806
    # Select nonoutlying chunklets and remove the discourse vector.
    q15, q85 = np.quantile(chunklet_size, [0.15, 0.85])
    nonoutlying_chunklets = (q15 <= chunklet_size) & (chunklet_size <= q85)
    if np.any(nonoutlying_chunklets):
        discourse = np.mean(X[nonoutlying_chunklets, :], axis=0)
        discourse = discourse / np.linalg.norm(discourse)
        X_modulo = X - np.outer(X @ discourse, discourse)  # noqa: N806
        if not np.any(np.linalg.norm(X_modulo, axis=1) <= np.finfo(X.dtype).eps):
            X = X_modulo  # noqa: N806
            X = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806
    # For each partition point in the list of chunklets, compute the similarity of chunklet before
    # and after the partition point.
    partition_similarity = np.sum(X[:-1] * X[1:], axis=1)
    # Make partition similarity nonnegative before modification and optimisation.
    partition_similarity = np.maximum(
        (partition_similarity + 1) / 2, np.sqrt(np.finfo(X.dtype).eps)
    )
    # Modify the partition similarity to encourage splitting on Markdown headings.
    prev_chunklet_is_heading = True
    for i, chunklet in enumerate(chunklets[:-1]):
        is_heading = bool(re.match(r"^#+\s", chunklet.replace("\n", "").strip()))
        if is_heading:
            # Encourage splitting before a heading.
            if not prev_chunklet_is_heading:
                partition_similarity[i - 1] = partition_similarity[i - 1] / 4
            # Don't split immediately after a heading.
            partition_similarity[i] = 1.0
        prev_chunklet_is_heading = is_heading
    # Solve an optimisation problem to find the best partition points.
    chunklet_size_cumsum = np.cumsum(chunklet_size)
    row_indices = []
    col_indices = []
    data = []
    for i in range(len(chunklets) - 1):
        r = chunklet_size_cumsum[i - 1] if i > 0 else 0
        idx = np.searchsorted(chunklet_size_cumsum - r, max_size, side="right")
        assert idx > i
        if idx == len(chunklet_size_cumsum):
            break
        cols = list(range(i, idx))
        col_indices.extend(cols)
        row_indices.extend([i] * len(cols))
        data.extend([1] * len(cols))
    A = coo_matrix(  # noqa: N806
        (data, (row_indices, col_indices)),
        shape=(max(row_indices) + 1, len(chunklets) - 1),
        dtype=np.float32,
    )
    b_ub = np.ones(A.shape[0], dtype=np.float32)
    res = linprog(
        partition_similarity,
        A_ub=-A,
        b_ub=-b_ub,
        bounds=(0, 1),
        integrality=[1] * A.shape[1],
    )
    if not res.success:
        error_message = "Optimization of chunk partitions failed."
        raise ValueError(error_message)
    # Split the chunklets and their window embeddings into optimal chunks.
    partition_indices = (np.where(res.x)[0] + 1).tolist()
    chunks = [
        "".join(chunklets[i:j])
        for i, j in zip([0, *partition_indices], [*partition_indices, len(chunklets)], strict=True)
    ]
    chunk_embeddings = np.split(chunklet_embeddings, partition_indices)
    return chunks, chunk_embeddings
