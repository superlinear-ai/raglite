"""Tests for the _insert module."""

from sqlmodel import Session, select

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Document, create_database_engine
from raglite._insert import insert_document

# --- Constants ---
TEST_FILENAME = "test_markdown_insertion.md"
MARKDOWN_INPUT = """This is some text before the first heading.
# Test the chunking
I want to ensure that the markdown header is correctly identified for the first chunk in a new section
## This is a subheading
What heading will this sub-paragraph get? It remains interesting...
# The second test of the chunking
Does the header actually have the right markdown header identified? or is it Test the chunking?
# Third test
What about this chunk? Does it get the right heading?
"""

# Define expected heading/body start combinations clearly
EXPECTED_COMBOS = [
    (
        "# Test the chunking",
        "I want to ensure that",  # Start of the first chunk's body in this section
    ),
    (
        "# Test the chunking\n## This is a subheading",
        "What heading will",  # Start of the second chunk's body
    ),
    (
        "# The second test of the chunking",
        "Does the header actually",  # Start of the third chunk's body
    ),
    (
        "# Third test",
        "What about this chunk?",  # Start of the fourth chunk's body (assuming it gets chunked)
    ),
]


# --- Test Function ---
def test_insert_document_from_string(raglite_test_config: RAGLiteConfig) -> None:
    """Test inserting a document from a Markdown string."""
    # Create a test configuration that uses the same database as raglite_test_config
    config = RAGLiteConfig(
        db_url=raglite_test_config.db_url,
        llm=raglite_test_config.llm,
        embedder=raglite_test_config.embedder,
        chunk_max_size=50,
    )

    # Insert the document
    insert_document(MARKDOWN_INPUT, filename=TEST_FILENAME, config=config)

    # Check that the document was inserted correctly
    engine = create_database_engine(config)
    with Session(engine) as session:
        # Query for the document
        document = session.exec(select(Document).where(Document.filename == TEST_FILENAME)).first()
        assert document is not None

        # Check that chunks were created
        chunks = session.exec(select(Chunk).where(Chunk.document_id == document.id)).all()
        assert len(chunks) > 0, "Chunks should have been created for the document."

        # Check that the first body does not have a heading
        assert chunks[0].body.strip().startswith("#") is False

        # Check that bodies do not contain headings
        assert all(chunk.body.strip().startswith("#") is False for chunk in chunks)

        # Look up the chunks that start with the expected bodies, check if the heading is correct
        for expected_heading, expected_body_start in EXPECTED_COMBOS:
            # Find the chunk that starts with the expected body text
            target_chunk = next(
                (c for c in chunks if c.body.strip().startswith(expected_body_start)),
                None,  # Return None if not found
            )

            # Assert that the chunk was found
            assert target_chunk is not None, (
                f"Could not find chunk starting with: '{expected_body_start}'"
            )

            # Assert that the found chunk has the correct heading
            assert target_chunk.headings.strip() == expected_heading, (
                f"Mismatch for chunk starting with '{expected_body_start}':\n"
                f"  Expected heading: '{expected_heading}'\n"
                f"  Actual heading:   '{target_chunk.headings.strip()}'"
            )
