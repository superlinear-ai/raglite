"""Split a document into chunklets."""

from collections.abc import Callable

import numpy as np
from markdown_it import MarkdownIt

from raglite._typing import FloatVector


def markdown_chunklet_boundaries(sentences: list[str]) -> FloatVector:
    """Estimate chunklet boundary probabilities given a Markdown document."""
    # Parse the document.
    doc = "".join(sentences)
    md = MarkdownIt()
    tokens = md.parse(doc)
    # Identify the character index of each line in the document.
    lines = doc.splitlines(keepends=True)
    line_start_char = [0]
    for line in lines[:-1]:
        line_start_char.append(line_start_char[-1] + len(line))
    # Identify the character index of each sentence in the document.
    sentence_start_char = [0]
    for sentence in sentences:
        sentence_start_char.append(sentence_start_char[-1] + len(sentence))
    # Map each line index to a corresponding sentence index.
    line_to_sentence = np.searchsorted(sentence_start_char, line_start_char, side="right") - 1
    # Configure probabilities for token types to be chunklet boundaries.
    token_type_to_proba = {
        "blockquote_open": 0.75,
        "bullet_list_open": 0.25,
        "heading_open": 1.0,
        "paragraph_open": 0.5,
        "ordered_list_open": 0.25,
    }
    # Compute the boundary probabilities for each sentence.
    last_sentence = -1
    boundary_probas = np.zeros(len(sentences))
    for token in tokens:
        if token.type in token_type_to_proba:
            start_line, _ = token.map  # type: ignore[misc]
            if (i := line_to_sentence[start_line]) != last_sentence:
                boundary_probas[i] = token_type_to_proba[token.type]
                last_sentence = i  # type: ignore[assignment]
    # For segments of consecutive boundaries, encourage splitting on the largest boundary in the
    # segment by setting the other boundary probabilities in the segment to zero.
    mask = boundary_probas != 0.0
    split_indices = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    segments = np.split(boundary_probas, split_indices)
    for segment in segments:
        max_idx, max_proba = np.argmax(segment), np.max(segment)
        segment[:] = 0.0
        segment[max_idx] = max_proba
    boundary_probas = np.concatenate(segments)
    return boundary_probas


def compute_num_statements(sentences: list[str]) -> FloatVector:
    """Compute the approximate number of statements of each sentence in a list of sentences."""
    sentence_word_length = np.asarray(
        [len(sentence.split()) for sentence in sentences], dtype=np.float64
    )
    q25, q75 = np.quantile(sentence_word_length, [0.25, 0.75])
    q25 = max(q25, np.sqrt(np.finfo(np.float64).eps))
    q75 = max(q75, q25 + np.sqrt(np.finfo(np.float64).eps))
    num_statements = np.piecewise(
        sentence_word_length,
        [sentence_word_length <= q25, sentence_word_length > q25],
        [lambda n: 0.75 * n / q25, lambda n: 0.75 + 0.5 * (n - q25) / (q75 - q25)],
    )
    return num_statements


def split_chunklets(
    sentences: list[str],
    boundary_cost: Callable[[FloatVector], float] = lambda p: (1.0 - p[0]) + np.sum(p[1:]),
    statement_cost: Callable[[float], float] = lambda s: ((s - 3) ** 2 / np.sqrt(max(s, 1e-6)) / 2),
    max_size: int = 2048,
) -> list[str]:
    """Split sentences into optimal chunklets.

    A chunklet is a concatenated contiguous list of sentences from a document. This function
    optimally partitions a document into chunklets using dynamic programming.

    A chunklet is considered optimal when it contains as close to 3 statements as possible, when the
    first sentence in the chunklet is a Markdown boundary such as the start of a heading or
    paragraph, and when the remaining sentences in the chunklet are not Markdown boundaries.

    Here, we define the number of statements in a sentence as a measure of the sentence's
    information content. A sentence is said to contain 1 statement if it contains the median number
    of words per sentence, across sentences in the document.

    The given document of sentences is optimally partitioned into chunklets by solving a dynamic
    programming problem that assigns a cost to each chunklet given the
    `boundary_cost(boundary_probas)` function and the `statement_cost(num_statements)` function. The
    former outputs the cost associated with the boundaries of the chunklet's sentences given the
    sentences' Markdown boundary probabilities, while the latter outputs the cost of the total
    number of statements in the chunklet.

    Parameters
    ----------
    sentences
        The input document as a list of sentences.
    boundary_cost
        A function that computes the boundary cost of a chunklet given the boundary probabilities of
        its sentences. The total cost of a chunklet is the sum of its boundary and statement cost.
    statement_cost
        A function that computes the statement cost of a chunklet given its number of statements.
        The total cost of a chunklet is the sum of its boundary and statement cost.
    max_size
        The maximum size of a chunklet in characters.

    Returns
    -------
    list[str]
        The document optimally partitioned into chunklets.
    """
    # Precompute chunklet boundary probabilities and each sentence's number of statements.
    boundary_probas = markdown_chunklet_boundaries(sentences)
    num_statements = compute_num_statements(sentences)
    # Initialize a dynamic programming table and backpointers. The dynamic programming table dp[i]
    # is defined as the minimum cost to segment the first i sentences (i.e., sentences[:i]).
    num_sentences = len(sentences)
    dp = np.full(num_sentences + 1, np.inf)
    dp[0] = 0.0
    back = -np.ones(num_sentences + 1, dtype=np.intp)
    # Compute the cost of partitioning sentences into chunklets.
    for i in range(1, num_sentences + 1):
        for j in range(i):
            # Limit the chunklets to a maximum size.
            if sum(len(s) for s in sentences[j:i]) > max_size:
                continue
            # Compute the cost of partitioning sentences[j:i] into a single chunklet.
            cost_ji = boundary_cost(boundary_probas[j:i])
            cost_ji += statement_cost(np.sum(num_statements[j:i]))
            # Compute the cost of partitioning sentences[:i] if we were to split at j.
            cost_0i = dp[j] + cost_ji
            # If the cost is less than the current minimum, update the DP table and backpointer.
            if cost_0i < dp[i]:
                dp[i] = cost_0i
                back[i] = j
    # Recover the optimal partitioning.
    partition_indices: list[int] = []
    i = back[num_sentences]
    while i > 0:
        partition_indices.append(i)
        i = back[i]
    partition_indices.reverse()
    # Split the sentences into optimal chunklets.
    chunklets = [
        "".join(sentences[i:j])
        for i, j in zip([0, *partition_indices], [*partition_indices, len(sentences)], strict=True)
    ]
    return chunklets
