"""Sentence splitter."""

import re
import warnings
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from markdown_it import MarkdownIt
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog
from wtpsplit_lite import SaT
from wtpsplit_lite._utils import indices_to_sentences

from raglite._typing import FloatVector


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


@cache
def _load_sat() -> tuple[SaT, dict[str, Any]]:
    """Load a Segment any Text (SaT) model."""
    sat = SaT("sat-3l-sm")  # This model makes the best trade-off between speed and accuracy.
    sat_kwargs = {"stride": 128, "block_size": 256, "weighting": "hat"}
    return sat, sat_kwargs


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
        The sentences.
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
    # Solve an optimisation problem to find the best sentence boundaries given the predicted
    # boundary probabilities. The objective is to select boundaries that maximise the sum of the
    # boundary probabilities above a given threshold, subject to the resulting sentences not being
    # smaller than the given minimum length.
    sentence_threshold = 0.25  # Default threshold for -sm models.
    c = probas - sentence_threshold
    N = len(probas)  # noqa: N806
    M = N - min_len + 1  # noqa: N806
    diagonals = [np.ones(M, dtype=np.float32) for _ in range(min_len)]
    offsets = list(range(min_len))
    A_ub = sparse.diags(diagonals, offsets, shape=(M, N), format="csr")  # noqa: N806
    b_ub = np.ones(M, dtype=np.float32)
    x0 = (probas >= sentence_threshold).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)  # Ignore x0 not being used.
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), x0=x0, integrality=[1] * N)
    if not res.success:
        error_message = "Optimization of sentence partitions failed."
        raise ValueError(error_message)
    # Split the document into sentences where the boundary probability exceeds a threshold.
    # TODO: Embed the trailing whitespace logic of `indices_to_sentences` in the optimization model.
    partition_indices = np.where(res.x > 0.5)[0]  # noqa: PLR2004
    sentences: list[str] = [s for s in indices_to_sentences(doc, partition_indices) if s.strip()]  # type: ignore[no-untyped-call]
    # Apply additional splits on paragraphs and sentences.
    if max_len is not None:
        for pattern in (r"(?<=\n\n)", r"(?<=\.\s)"):
            sentences = [
                part
                for sent in sentences
                for part in ([sent] if len(sent) <= max_len else re.split(pattern, sent))
            ]
    # Recursively split long sentences in the middle if they are still too long.
    if max_len is not None:
        while any(len(sentence) > max_len for sentence in sentences):
            sentences = [
                part
                for sent in sentences
                for part in (
                    [sent]
                    if len(sent) <= max_len
                    else [sent[: len(sent) // 2], sent[len(sent) // 2 :]]
                )
            ]
    return sentences
