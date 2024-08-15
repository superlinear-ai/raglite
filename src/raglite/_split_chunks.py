"""Split a document into semantic chunks."""

import re
from collections.abc import Callable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from raglite._embed import embed_strings
from raglite._typing import FloatMatrix


def split_chunks(
    sentences: list[str],
    max_size: int = 1440,
    sentence_window_size: int = 3,
    embed: Callable[[list[str]], FloatMatrix] = embed_strings,
) -> tuple[list[str], list[FloatMatrix]]:
    """Split sentences into optimal semantic chunks."""
    # Window the sentences.
    whisker_size = (sentence_window_size - 1) // 2
    windows = [
        "".join(sentences[max(0, i - whisker_size) : min(i + whisker_size + 1, len(sentences))])
        for i in range(len(sentences))
    ]
    window_embeddings = embed(windows)
    # Normalise the sentence embeddings to unit norm.
    window_embeddings = window_embeddings / np.linalg.norm(window_embeddings, axis=1, keepdims=True)
    # Select nonoutlying sentences.
    window_length = np.asarray([len(window) for window in windows])
    q15, q85 = np.quantile(window_length, [0.15, 0.85])
    outlying_windows = (window_length <= q15) | (q85 <= window_length)
    # Remove the global discourse vector.
    X = window_embeddings.copy()  # noqa: N806
    discourse = np.mean(X[~outlying_windows, :], axis=0)
    X = X - np.outer(X @ discourse, discourse)  # noqa: N806
    # Renormalise to unit norm.
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806
    # For each partition point in the list of sentences, compute the similarity of the windows
    # before and after the partition point.
    windows_before = X[: -(1 + 2 * whisker_size)]
    windows_after = X[1 + 2 * whisker_size :]
    partition_similarity = np.ones(len(sentences) - 1, dtype=X.dtype)
    partition_similarity[whisker_size:-whisker_size] = np.sum(
        windows_before * windows_after, axis=1
    )
    # Make partition similarity nonnegative before modification and optimisation.
    partition_similarity = 1e-4 + (partition_similarity + 1) / 2
    # Modify the partition similarity to encourage splitting on Markdown headings.
    prev_sentence_is_heading = True
    for i, sentence in enumerate(sentences[:-1]):
        is_heading = bool(re.match(r"^#+\s", sentence.replace("\n", "").strip()))
        if is_heading:
            # Encourage splitting before a heading.
            if not prev_sentence_is_heading:
                partition_similarity[i - 1] = partition_similarity[i - 1] / 4
            # Don't split immediately after a heading.
            partition_similarity[i] = 1.0
        prev_sentence_is_heading = is_heading
    # Solve an optimisation problem to find the best partition points.
    sentence_length = np.asarray([len(sentence) for sentence in sentences])
    sentence_length_cumsum = np.cumsum(sentence_length)
    row_indices = []
    col_indices = []
    data = []
    for i in range(len(sentences) - 1):
        r = sentence_length_cumsum[i - 1] if i > 0 else 0
        idx = np.searchsorted(sentence_length_cumsum - r, max_size)
        if idx == i:
            error_message = "Sentence with length larger than chunk max_size detected."
            raise ValueError(error_message)
        if idx == len(sentence_length_cumsum):
            break
        cols = list(range(i, idx))
        col_indices.extend(cols)
        row_indices.extend([i] * len(cols))
        data.extend([1] * len(cols))
    A = coo_matrix(  # noqa: N806
        (data, (row_indices, col_indices)),
        shape=(max(row_indices) + 1, len(sentences) - 1),
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
    assert res.success, "Optimization of chunk partitions failed."
    # Split the sentences and their window embeddings into optimal chunks.
    partition_indices = (np.where(res.x)[0] + 1).tolist()
    chunks = [
        "".join(sentences[i:j])
        for i, j in zip([0, *partition_indices], [*partition_indices, len(sentences)], strict=True)
    ]
    multi_vector_embeddings = [
        window_embeddings[i:j]
        for i, j in zip([0, *partition_indices], [*partition_indices, len(sentences)], strict=True)
    ]
    return chunks, multi_vector_embeddings
