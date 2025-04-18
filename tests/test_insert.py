"""Tests for the _insert module."""

from sqlmodel import Session, select

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Document, create_database_engine
from raglite._insert import insert_document

TEST_FILENAME = "test_markdown_insertion.md"
MARKDOWN_INPUT = """This is some text before the first heading.
# Test the chunking
I want to ensure that the markdown header is correctly identified for the first chunk in a new section
## This is a subheading
What heading will this sub-paragraph get? It remains interesting...
## The second subheading
What happens when a second sub-heading arrives? Will it overwrite the first sub-heading?
### A sub-sub heading
This paragraph contains text under a sub-sub-heading.
# The second test of the chunking
Does the header actually have the right markdown header identified? or is it Test the chunking?
# Third test
What about this chunk? Does it get the right heading?
"""


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

        restored_document = ""
        for chunk in chunks:
            # body should not contain the heading string (except if heading is empty)
            if chunk.headings.strip() != "":
                assert chunk.headings.strip() not in chunk.body.strip(), (
                    f"Chunk body contains heading: '{chunk.headings.strip()}'\n"
                    f"Chunk body: '{chunk.body.strip()}'"
                )

            # Body that starts with a # should not have a heading
            if chunk.body.strip().startswith("# "):
                assert chunk.headings.strip() == "", (
                    f"Chunk body starts with a heading: '{chunk.body.strip()}'\n"
                    f"Chunk headings: '{chunk.headings.strip()}'"
                )

            restored_document += chunk.body

        # combining the chunks should give the original document
        restored_document = "".join(restored_document)
        restored_document = restored_document.replace("\n", "").strip()
        markdown_input = MARKDOWN_INPUT.replace("\n", "").strip()
        assert restored_document == markdown_input, (
            "Restored document does not match the original input."
        )
