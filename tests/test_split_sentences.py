"""Test RAGLite's sentence splitting functionality."""

from pathlib import Path

from raglite._markdown import document_to_markdown
from raglite._split_sentences import split_sentences


def test_split_sentences() -> None:
    """Test splitting a document into sentences."""
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    doc = document_to_markdown(doc_path)
    sentences = split_sentences(doc)
    expected_sentences = [
        "# ON THE ELECTRODYNAMICS OF MOVING BODIES\n\n",
        "## By A. EINSTEIN June 30, 1905\n\n",
        "It is known that Maxwell’s electrodynamics—as usually understood at the\npresent time—when applied to moving bodies, leads to asymmetries which do\n\nnot appear to be inherent in the phenomena. ",  # noqa: RUF001
        "Take, for example, the recipro-\ncal electrodynamic action of a magnet and a conductor. ",
        "The observable phe-\nnomenon here depends only on the relative motion of the conductor and the\nmagnet, whereas the customary view draws a sharp distinction between the two\ncases in which either the one or the other of these bodies is in motion. ",
        "For if the\nmagnet is in motion and the conductor at rest, there arises in the neighbour-\nhood of the magnet an electric field with a certain definite energy, producing\na current at the places where parts of the conductor are situated. ",
        "But if the\n\nmagnet is stationary and the conductor in motion, no electric field arises in the\nneighbourhood of the magnet. ",
        "In the conductor, however, we find an electro-\nmotive force, to which in itself there is no corresponding energy, but which gives\nrise—assuming equality of relative motion in the two cases discussed—to elec-\n\ntric currents of the same path and intensity as those produced by the electric\nforces in the former case.\n\n",
        "Examples of this sort, together with the unsuccessful attempts to discover\nany motion of the earth relatively to the “light medium,” suggest that the\n\nphenomena of electrodynamics as well as of mechanics possess no properties\ncorresponding to the idea of absolute rest. ",
        "They suggest rather that, as has\nalready been shown to the first order of small quantities, the same laws of\nelectrodynamics and optics will be valid for all frames of reference for which the\nequations of mechanics hold good.1 ",
        "We will raise this conjecture (the purport\nof which will hereafter be called the “Principle of Relativity”) to the status\n\nof a postulate, and also introduce another postulate, which is only apparently\nirreconcilable with the former, namely, that light is always propagated in empty\nspace with a definite velocity c which is independent of the state of motion of the\nemitting body. ",
        "These two postulates suffice for the attainment of a simple and\nconsistent theory of the electrodynamics of moving bodies based on Maxwell’s\ntheory for stationary bodies. ",  # noqa: RUF001
        "The introduction of a “luminiferous ether” will\nprove to be superfluous inasmuch as the view here to be developed will not\nrequire an “absolutely stationary space” provided with special properties, nor\n",
        "1The preceding memoir by Lorentz was not at this time known to the author.\n\n",
        "assign a velocity-vector to a point of the empty space in which electromagnetic\nprocesses take place.\n\n",
        "The theory to be developed is based—like all electrodynamics—on the kine-\nmatics of the rigid body, since the assertions of any such theory have to do\nwith the relationships between rigid bodies (systems of co-ordinates), clocks,\nand electromagnetic processes. ",
        "Insufficient consideration of this circumstance\nlies at the root of the difficulties which the electrodynamics of moving bodies\nat present encounters.\n\n",
        "## I. KINEMATICAL PART § **1. Definition of Simultaneity**\n\n",
        "Let us take a system of co-ordinates in which the equations of Newtonian\nmechanics hold good.2 ",
    ]
    assert isinstance(sentences, list)
    assert all(
        sentence == expected_sentence
        for sentence, expected_sentence in zip(
            sentences[: len(expected_sentences)], expected_sentences, strict=True
        )
    )
