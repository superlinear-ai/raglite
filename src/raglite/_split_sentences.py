"""Split a document into sentences."""

from collections import deque
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from markdown_it import MarkdownIt
from wtpsplit_lite import SaT

from raglite._typing import FloatVector


@cache
def _load_sat() -> tuple[SaT, dict[str, Any]]:
    """Load a Segment any Text (SaT) model."""
    sat = SaT("sat-1l-sm")  # This model makes the best trade-off between speed and accuracy.
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
        line_start_char = [0]
        for line in lines:
            line_start_char.append(line_start_char[-1] + len(line))
        for token in tokens:
            if token.type == "heading_open":
                start_line, end_line = token.map  # type: ignore[misc]
                heading_start = line_start_char[start_line]
                heading_end = line_start_char[end_line]
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


def _split_sentences(  # noqa: C901, PLR0912, PLR0915
    doc: str, probas: FloatVector, *, min_len: int, max_len: int | None = None
) -> list[str]:
    """Find optimal sentence boundaries using dynamic programming.

    Maximises the sum of the boundary probabilities above a threshold, subject to the resulting
    sentences not being shorter or longer than the given minimum or maximum length, respectively.

    A boundary at position i means the character at index i is the last character of a sentence
    (the next sentence starts at i + 1).
    """
    sentence_threshold = 0.25  # Default threshold for -sm models.
    n = len(probas)
    first_valid = min_len - 1  # Earliest boundary ensuring first sentence has >= min_len chars.
    last_valid = n - min_len - 1  # Latest boundary ensuring last sentence has >= min_len chars.
    if last_valid < first_valid:
        return [doc]
    # Score for placing a boundary at each position: positive means "good boundary".
    scores = probas - sentence_threshold
    # dp[i] = maximum total score with the last boundary at position i.
    # back[i] = position of the previous boundary (-1 means "first boundary", no predecessor).
    dp = np.full(n, -np.inf)
    back = np.full(n, -1, dtype=np.intp)
    if max_len is None:
        # No maximum length: use a running maximum for O(N) computation.
        best_prev = -np.inf
        best_prev_idx = -1
        for i in range(first_valid, last_valid + 1):
            # Position i - min_len becomes a valid predecessor for position i.
            j = i - min_len
            if j >= first_valid and dp[j] > best_prev:
                best_prev = dp[j]
                best_prev_idx = j
            # Option 1: this is the first (and possibly only) boundary.
            dp[i] = scores[i]
            # Option 2: extend from the best previous boundary.
            if best_prev > -np.inf and best_prev + scores[i] > dp[i]:
                dp[i] = best_prev + scores[i]
                back[i] = best_prev_idx
    else:
        # With maximum length: use a monotonic deque for O(N) sliding window maximum.
        dq: deque[tuple[float, int]] = deque()
        for i in range(first_valid, last_valid + 1):
            # Position i - min_len becomes a valid predecessor for position i.
            j = i - min_len
            if j >= first_valid and np.isfinite(dp[j]):
                while dq and dq[-1][0] <= dp[j]:
                    dq.pop()
                dq.append((dp[j], j))
            # Evict predecessors that would create a sentence longer than max_len.
            while dq and dq[0][1] < i - max_len:
                dq.popleft()
            # Option 1: this is the first boundary (sentence is doc[0:i+1]).
            if i + 1 <= max_len:
                dp[i] = scores[i]
            # Option 2: extend from the best previous boundary in the valid window.
            if dq and dq[0][0] + scores[i] > dp[i]:
                dp[i] = dq[0][0] + scores[i]
                back[i] = dq[0][1]
    # Find the best final boundary such that the trailing sentence is also valid.
    answer_min = first_valid
    if max_len is not None:
        answer_min = max(answer_min, n - max_len - 1)
    no_boundary_valid = max_len is None or max_len >= n
    best_score = 0.0 if no_boundary_valid else -np.inf
    best_last = -1
    for i in range(answer_min, last_valid + 1):
        if dp[i] > best_score:
            best_score = dp[i]
            best_last = i
    if best_last == -1:
        if no_boundary_valid:
            return [doc]
        error_message = "Sentence partition failed: no valid split satisfies the constraints."
        raise ValueError(error_message)
    # Backtrack to recover the optimal boundary positions.
    boundaries: list[int] = []
    pos = best_last
    while pos >= 0:
        boundaries.append(pos)
        pos = back[pos]
    boundaries.reverse()
    # Split the document at the boundary positions.
    partition_indices = [b + 1 for b in boundaries]
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
        The minimum number of characters in a sentence.
    max_len
        The maximum number of characters in a sentence, with no maximum by default.
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
        result_sentences: list[str] = []
        pos = 0
        for sentence in sentences:
            if len(sentence) <= max_len:
                result_sentences.append(sentence)
            else:
                result_sentences.extend(
                    _split_sentences(
                        sentence,
                        probas[pos : pos + len(sentence)],
                        min_len=min_len,
                        max_len=max_len,
                    )
                )
            pos += len(sentence)
        sentences = result_sentences
    return sentences
