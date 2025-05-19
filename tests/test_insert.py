"""Test RAGLite's document insertion."""

from pathlib import Path

from sqlmodel import Session, select
from tqdm import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Document, create_database_engine
from raglite._markdown import document_to_markdown


def test_insert(raglite_test_config: RAGLiteConfig) -> None:
    """Test document insertion."""
    # Get access to the database from the raglite_test_config
    engine = create_database_engine(raglite_test_config)
    # Open a session to extract document and chunks from the existing database
    with Session(engine) as session:
        # Get the first document from the database (already inserted by the fixture)
        document = session.exec(select(Document)).first()
        assert document is not None, "No document found in the database"
        # Get the existing chunks for this document
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.index)  # type: ignore[arg-type]
        ).all()
        assert len(chunks) > 0, "No chunks found for the document"
        restored_document = ""
        for chunk in tqdm(chunks, desc="Processing chunks"):
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
