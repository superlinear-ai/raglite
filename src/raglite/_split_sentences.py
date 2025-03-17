"""Split a document into sentences."""

import warnings
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from markdown_it import MarkdownIt
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog
from wtpsplit_lite import SaT

from raglite._typing import FloatVector


@cache
def _load_sat() -> tuple[SaT, dict[str, Any]]:
    """Load a Segment any Text (SaT) model."""
    sat = SaT("sat-3l-sm")  # This model makes the best trade-off between speed and accuracy.
    sat_kwargs = {"stride": 128, "block_size": 256, "weighting": "hat"}
    return sat, sat_kwargs


def markdown_sentence_boundaries(doc: str) -> FloatVector:
    """Determine known sentence boundaries from a Markdown document."""

    def get_markdown_heading_indexes(doc: str) -> list[tuple[int, int]]:
        """Get the indexes of the headings in a Markdown document."""
        md = MarkdownIt()
        tokens = md.parse(doc)
        headings = []
        lines = doc.splitlines(keepends=True)
        char_idx = [0]
        for line in lines:
            char_idx.append(char_idx[-1] + len(line))
        for token in tokens:
            if token.type == "heading_open":
                start_line, end_line = token.map  # type: ignore[misc]
                heading_start = char_idx[start_line]
                heading_end = char_idx[end_line]
                headings.append((heading_start, heading_end + 1))
        return headings

    # Get the start and end character indexes of the headings in the document.
    headings = get_markdown_heading_indexes(doc)
    # Indicate that each heading is a contiguous sentence by setting the boundary probabilities.
    boundary_probas = np.full(len(doc), np.nan)
    for heading_start, heading_end in headings:
        if 0 <= heading_start - 1 < len(boundary_probas):
            boundary_probas[heading_start - 1] = 1  # First heading character starts a sentence.
        boundary_probas[heading_start : heading_end - 1] = 0  # Body does not contain boundaries.
        if 0 <= heading_end - 1 < len(boundary_probas):
            boundary_probas[heading_end - 1] = 1  # Last heading character is the end of a sentence.
    return boundary_probas


def _split_sentences(
    doc: str, probas: FloatVector, *, min_len: int, max_len: int | None = None
) -> list[str]:
    # Solve an optimisation problem to find the best sentence boundaries given the predicted
    # boundary probabilities. The objective is to select boundaries that maximise the sum of the
    # boundary probabilities above a given threshold, subject to the resulting sentences not being
    # shorter or longer than the given minimum or maximum length, respectively.
    sentence_threshold = 0.25  # Default threshold for -sm models.
    c = probas - sentence_threshold
    N = len(probas)  # noqa: N806
    M = N - min_len + 1  # noqa: N806
    diagonals = [np.ones(M, dtype=np.float32) for _ in range(min_len)]
    offsets = list(range(min_len))
    A_min = sparse.diags(diagonals, offsets, shape=(M, N), format="csr")  # noqa: N806
    b_min = np.ones(M, dtype=np.float32)
    bounds = [(0, 1)] * N
    bounds[: min_len - 1] = [(0, 0)] * (min_len - 1)  # Prevent short leading sentences.
    bounds[-min_len:] = [(0, 0)] * min_len  # Prevent short trailing sentences.
    if max_len is not None and (M := N - max_len + 1) > 0:  # noqa: N806
        diagonals = [np.ones(M, dtype=np.float32) for _ in range(max_len)]
        offsets = list(range(max_len))
        A_max = sparse.diags(diagonals, offsets, shape=(M, N), format="csr")  # noqa: N806
        b_max = np.ones(M, dtype=np.float32)
        A_ub = sparse.vstack([A_min, -A_max], format="csr")  # noqa: N806
        b_ub = np.hstack([b_min, -b_max])
    else:
        A_ub = A_min  # noqa: N806
        b_ub = b_min
    x0 = (probas >= sentence_threshold).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)  # Ignore x0 not being used.
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, x0=x0, integrality=[1] * N)
    if not res.success:
        error_message = "Optimization of sentence partitions failed."
        raise ValueError(error_message)
    # Split the document into sentences where the boundary probability exceeds a threshold.
    partition_indices = np.where(res.x > 0.5)[0] + 1  # noqa: PLR2004
    sentences = [
        doc[i:j] for i, j in zip([0, *partition_indices], [*partition_indices, None], strict=True)
    ]
    return sentences


def split_sentences(
    doc: str,
    min_len: int = 4,
    max_len: int | None = None,
    boundary_probas: FloatVector | Callable[[str], FloatVector] = markdown_sentence_boundaries,
) -> list[str]:
    """Split a document into sentences.

    Parameters
    ----------
    doc
        The document to split into sentences.
    min_len
        The minimum length of a sentence.
    max_len
        The maximum length of a sentence, with no maximum length by default.
    boundary_probas
        Any known sentence boundary probabilities to override the model's predicted sentence
        boundary probabilities. If an element of the probability vector with index k is 1 (0), then
        the character at index k + 1 is (not) the start of a sentence. Elements set to NaN will not
        override the predicted probabilities. By default, the known sentence boundary probabilities
        are determined from the document's Markdown headings.

    Returns
    -------
    list[str]
        The input document partitioned into sentences. All sentences are constructed to contain at
        least one non-whitespace character, not have any leading whitespace (except for the first
        sentence if the document itself has leading whitespace), and respect the minimum and maximum
        sentence lengths.
    """
    # Exit early if there is only one sentence to return.
    if len(doc) <= min_len:
        return [doc]
    # Compute the sentence boundary probabilities with a wtpsplit Segment any Text (SaT) model.
    sat, sat_kwargs = _load_sat()
    predicted_probas = sat.predict_proba(doc, **sat_kwargs)
    # Override the predicted boundary probabilities with the known boundary probabilities.
    known_probas = boundary_probas(doc) if callable(boundary_probas) else boundary_probas
    probas = predicted_probas.copy()
    probas[np.isfinite(known_probas)] = known_probas[np.isfinite(known_probas)]
    # Propagate the boundary probabilities so that whitespace is always trailing and never leading.
    is_space = np.array([char.isspace() for char in doc], dtype=np.bool_)
    start = np.where(np.insert(~is_space[:-1] & is_space[1:], len(is_space) - 1, False))[0]
    end = np.where(np.insert(~is_space[1:] & is_space[:-1], 0, False))[0]
    start = start[start < np.max(end, initial=-1)]
    end = end[end > np.min(start, initial=len(is_space))]
    for i, j in zip(start, end, strict=True):
        min_proba, max_proba = np.min(probas[i:j]), np.max(probas[i:j])
        probas[i : j - 1] = min_proba  # From the non-whitespace to the penultimate whitespace char.
        probas[j - 1] = max_proba  # The last whitespace char.
    # Solve an optimization problem to find optimal sentences with no maximum length. We delay the
    # maximum length constraint to a subsequent step to avoid blowing up memory usage.
    sentences = _split_sentences(doc, probas, min_len=min_len, max_len=None)
    # For each sentence that exceeds the maximum length, solve the optimization problem again with
    # a maximum length constraint.
    if max_len is not None:
        sentences = [
            subsentence
            for sentence in sentences
            for subsentence in (
                [sentence]
                if len(sentence) <= max_len
                else _split_sentences(
                    sentence,
                    probas[doc.index(sentence) : doc.index(sentence) + len(sentence)],
                    min_len=min_len,
                    max_len=max_len,
                )
            )
        ]
    return sentences
