"""Split a document into semantic chunks."""

import re

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from raglite._typing import FloatMatrix


def split_chunks(  # noqa: C901, PLR0915
    sentences: list[str],
    sentence_embeddings: FloatMatrix,
    sentence_window_size: int = 3,
    max_size: int = 1440,
) -> tuple[list[str], list[FloatMatrix]]:
    """Split sentences into optimal semantic chunks with corresponding sentence embeddings."""
    # Validate the input.
    sentence_length = np.asarray([len(sentence) for sentence in sentences])
    if not np.all(sentence_length <= max_size):
        error_message = "Sentence with length larger than chunk max_size detected."
        raise ValueError(error_message)
    if not np.all(np.linalg.norm(sentence_embeddings, axis=1) > 0.0):
        error_message = "Sentence embeddings with zero norm detected."
        raise ValueError(error_message)
    # Exit early if there is only one chunk to return.
    if len(sentences) <= 1 or sum(sentence_length) <= max_size:
        return ["".join(sentences)] if sentences else sentences, [sentence_embeddings]
    # Normalise the sentence embeddings to unit norm.
    X = sentence_embeddings.astype(np.float32)  # noqa: N806
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806
    # Select nonoutlying sentences and remove the discourse vector.
    q15, q85 = np.quantile(sentence_length, [0.15, 0.85])
    nonoutlying_sentences = (q15 <= sentence_length) & (sentence_length <= q85)
    discourse = np.mean(X[nonoutlying_sentences, :], axis=0)
    discourse = discourse / np.linalg.norm(discourse)
    if not np.any(np.linalg.norm(X - discourse[np.newaxis, :], axis=1) <= np.finfo(X.dtype).eps):
        X = X - np.outer(X @ discourse, discourse)  # noqa: N806
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806
    # For each partition point in the list of sentences, compute the similarity of the windows
    # before and after the partition point. Sentence embeddings are assumed to be of the sentence
    # itself and at most the (sentence_window_size - 1) sentences that preceed it.
    sentence_window_size = min(len(sentences) - 1, sentence_window_size)
    windows_before = X[:-sentence_window_size]
    windows_after = X[sentence_window_size:]
    partition_similarity = np.ones(len(sentences) - 1, dtype=X.dtype)
    partition_similarity[: len(windows_before)] = np.sum(windows_before * windows_after, axis=1)
    # Make partition similarity nonnegative before modification and optimisation.
    partition_similarity = np.maximum(
        (partition_similarity + 1) / 2, np.sqrt(np.finfo(X.dtype).eps)
    )
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
    sentence_length_cumsum = np.cumsum(sentence_length)
    row_indices = []
    col_indices = []
    data = []
    for i in range(len(sentences) - 1):
        r = sentence_length_cumsum[i - 1] if i > 0 else 0
        idx = np.searchsorted(sentence_length_cumsum - r, max_size)
        assert idx > i
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
    if not res.success:
        error_message = "Optimization of chunk partitions failed."
        raise ValueError(error_message)
    # Split the sentences and their window embeddings into optimal chunks.
    partition_indices = (np.where(res.x)[0] + 1).tolist()
    chunks = [
        "".join(sentences[i:j])
        for i, j in zip([0, *partition_indices], [*partition_indices, len(sentences)], strict=True)
    ]
    chunk_embeddings = np.split(sentence_embeddings, partition_indices)
    return chunks, chunk_embeddings
