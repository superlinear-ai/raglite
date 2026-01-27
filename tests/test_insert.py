"""Test RAGLite's document insertion."""

from pathlib import Path

from sqlmodel import Session, select
from tqdm import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Document, create_database_engine
from raglite._insert import _get_database_metadata, insert_documents
from raglite._markdown import document_to_markdown


def test_insert(raglite_test_config: RAGLiteConfig) -> None:
    """Test document insertion."""
    # Open a session to extract document and chunks from the existing database.
    with Session(create_database_engine(raglite_test_config)) as session:
        # Get the first document from the database (already inserted by the fixture).
        document = session.exec(select(Document)).first()
        assert document is not None, "No document found in the database"
        # Get the existing chunks for this document.
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.index)  # type: ignore[arg-type]
        ).all()
        assert len(chunks) > 0, "No chunks found for the document"
        restored_document = ""
        for chunk in tqdm(chunks, desc="Processing chunks", leave=False):
            # Body should not contain the heading string (except if heading is empty).
            if chunk.headings.strip() != "":
                assert chunk.headings.strip() not in chunk.body.strip(), (
                    f"Chunk body contains heading: '{chunk.headings.strip()}'\n"
                    f"Chunk body: '{chunk.body.strip()}'"
                )
            # Body that starts with a # should not have a heading.
            if chunk.body.strip().startswith("# "):
                assert chunk.headings.strip() == "", (
                    f"Chunk body starts with a heading: '{chunk.body.strip()}'\n"
                    f"Chunk headings: '{chunk.headings.strip()}'"
                )
            restored_document += chunk.body
        # Combining the chunks should yield the original document.
        restored_document = restored_document.replace("\n", "").strip()
        doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
        doc = document_to_markdown(doc_path)
        doc = doc.replace("\n", "").strip()
        assert restored_document == doc, "Restored document does not match the original input."
        # Verify that the document metadata matches.
        metadata = _get_database_metadata(session)
        assert len(metadata) > 0, "No metadata found for the document"
        # Check that the metadata values match the original document metadata.
        for meta in metadata:
            assert meta.name in document.metadata_, (
                f"Metadata {meta.name} not found in document metadata"
            )
            for value in document.metadata_[meta.name]:
                assert value in meta.values, (
                    f"Metadata value '{value}' for '{meta.name}' not found in database metadata"
                )


def test_insert_reuse_document_instance(
    raglite_test_config: RAGLiteConfig,
) -> None:
    """Reuse a document instance across calls without errors."""
    isolated_config = RAGLiteConfig(
        db_url="duckdb:///:memory:",
        llm=raglite_test_config.llm,
        embedder=raglite_test_config.embedder,
    )
    doc = Document.from_text(
        content="Reuse instance test content.",
        url="http://example.com/reuse",
        filename="reuse_instance.html",
        id="reuse-instance-test",
    )
    insert_documents([doc], config=isolated_config)
    insert_documents([doc], config=isolated_config)

    with Session(create_database_engine(isolated_config)) as session:
        documents = session.exec(select(Document).where(Document.id == "reuse-instance-test")).all()
        assert len(documents) == 1


def test_insert_duplicate_documents_with_same_id(
    raglite_test_config: RAGLiteConfig,
) -> None:
    """De-duplicate incoming documents that share the same id."""
    isolated_config = RAGLiteConfig(
        db_url="duckdb:///:memory:",
        llm=raglite_test_config.llm,
        embedder=raglite_test_config.embedder,
    )
    doc1 = Document.from_text(
        content="Duplicate id test content.",
        url="http://example.com/duplicate",
        filename="duplicate.html",
        id="duplicate-id-test",
    )
    doc2 = Document.from_text(
        content="Duplicate id test content.",
        url="http://example.com/duplicate",
        filename="duplicate.html",
        id="duplicate-id-test",
    )
    insert_documents([doc1, doc2], config=isolated_config)

    with Session(create_database_engine(isolated_config)) as session:
        documents = session.exec(select(Document).where(Document.id == "duplicate-id-test")).all()
        assert len(documents) == 1
