"""Test RAGLite's document deletion."""

from sqlmodel import Session, SQLModel, col, select

from raglite._config import RAGLiteConfig
from raglite._database import (
    Chunk,
    ChunkEmbedding,
    Document,
    create_database_engine,
)
from raglite._delete import delete_documents, delete_documents_by_metadata
from raglite._insert import _get_database_metadata, insert_documents


def test_delete(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion."""
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1, author="Test Author", type="A")
    doc1_id = document1.id
    insert_documents([document1], config=raglite_test_config)

    with Session(create_database_engine(raglite_test_config)) as session:
        chunk_ids = session.exec(
            select(Chunk.id).where(col(Chunk.document_id).in_([doc1_id]))
        ).all()

    deleted_count = delete_documents([doc1_id], config=raglite_test_config)
    assert deleted_count == 1, f"Expected 1 document to be deleted, but got {deleted_count}"

    with Session(create_database_engine(raglite_test_config)) as session:
        assert session.exec(select(Document).where(Document.id == doc1_id)).first() is None, (
            "Document was not deleted"
        )
        # Check that other tables are deleted
        for table_name, table in SQLModel.metadata.tables.items():
            if "document_id" in table.c:
                assert (
                    session.exec(
                        select(table.c.document_id).where(table.c.document_id == doc1_id)
                    ).first()  # type: ignore[attr-defined]
                    is None
                ), f"{table_name} still contains data for deleted document"
        assert (
            session.exec(
                select(ChunkEmbedding).where(col(ChunkEmbedding.chunk_id).in_(chunk_ids))
            ).first()
            is None
        ), "Chunk embeddings were not deleted"
        existing_metadata = {
            record.name: record for record in _get_database_metadata(session=session)
        }
        assert "type" not in existing_metadata, "Metadata field 'type' was not deleted"
        assert "Test Author" not in existing_metadata["author"].values, (
            "Metadata field 'author' was not deleted"
        )


def test_delete_by_metadata(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion by metadata."""
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1, metadata="A")
    insert_documents([document1], config=raglite_test_config)
    document2 = Document.from_text(content1 + " diff", metadata="A")
    insert_documents([document2], config=raglite_test_config)

    deleted_count = delete_documents_by_metadata({"metadata": "A"}, config=raglite_test_config)
    assert deleted_count == 2, f"Expected 2 documents to be deleted, but got {deleted_count}"  # noqa: PLR2004
