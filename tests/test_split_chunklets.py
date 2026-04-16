"""Test RAGLite's chunk splitting functionality."""

import pytest

from raglite._split_chunklets import split_chunklets


@pytest.mark.parametrize(
    "sentences_splits",
    [
        pytest.param(
            (
                [
                    # Sentence 1:
                    "It is known that Maxwell’s electrodynamics—as usually understood at the\n"  # noqa: RUF001
                    "present time—when applied to moving bodies, leads to asymmetries which do\n\n"
                    "not appear to be inherent in the phenomena. ",
                    # Sentence 2:
                    "Take, for example, the recipro-\ncal electrodynamic action of a magnet and a conductor. \n\n",
                    # Sentence 3 (heading):
                    "# ON THE ELECTRODYNAMICS OF MOVING BODIES\n\n",
                    # Sentence 4 (heading):
                    "## By A. EINSTEIN June 30, 1905\n\n",
                    # Sentence 5 (paragraph boundary):
                    "The observable phe-\n"
                    "nomenon here depends only on the relative motion of the conductor and the\n"
                    "magnet, whereas the customary view draws a sharp distinction between the two\n"
                    "cases in which either the one or the other of these bodies is in motion. ",
                    # Sentence 6:
                    "For if the\n"
                    "magnet is in motion and the conductor at rest, there arises in the neighbour-\n"
                    "hood of the magnet an electric field with a certain definite energy, producing\n"
                    "a current at the places where parts of the conductor are situated. ",
                ],
                [2],
            ),
            id="consecutive_boundaries",
        ),
        pytest.param(
            (
                [
                    # Sentence 1:
                    "The theory to be developed is based—like all electrodynamics—on the kine-\n"
                    "matics of the rigid body, since the assertions of any such theory have to do\n"
                    "with the relationships between rigid bodies (systems of co-ordinates), clocks,\n"
                    "and electromagnetic processes. ",
                    # Sentence 2:
                    "Insufficient consideration of this circumstance\n"
                    "lies at the root of the difficulties which the electrodynamics of moving bodies\n"
                    "at present encounters.\n\n",
                    # Sentence 3 (paragraph boundary):
                    "The observable phe-\n"
                    "nomenon here depends only on the relative motion of the conductor and the\n"
                    "magnet, whereas the customary view draws a sharp distinction between the two\n"
                    "cases in which either the one or the other of these bodies is in motion. ",
                    # Sentence 4:
                    "For if the\n"
                    "magnet is in motion and the conductor at rest, there arises in the neighbour-\n"
                    "hood of the magnet an electric field with a certain definite energy, producing\n"
                    "a current at the places where parts of the conductor are situated. ",
                    # Sentence 5:
                    "But if the\n"
                    "magnet is stationary and the conductor in motion, no electric field arises in the\n"
                    "neighbourhood of the magnet. ",
                ],
                [2],
            ),
            id="paragraph_boundary",
        ),
    ],
)
def test_split_chunklets(sentences_splits: tuple[list[str], list[int]]) -> None:
    """Test chunklet splitting."""
    sentences, splits = sentences_splits
    chunklets = split_chunklets(sentences)
    expected_chunklets = [
        "".join(sentences[i:j])
        for i, j in zip([0, *splits], [*splits, len(sentences)], strict=True)
    ]
    assert isinstance(chunklets, list)
    assert all(isinstance(chunklet, str) for chunklet in chunklets)
    assert sum(len(chunklet) for chunklet in chunklets) == sum(
        len(sentence) for sentence in sentences
    )
    assert all(
        chunklet == expected_chunklet
        for chunklet, expected_chunklet in zip(chunklets, expected_chunklets, strict=True)
    )
