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
    document1 = Document.from_text(content1, author="Test Author", classification="A")
    doc1_id = document1.id
    insert_documents([document1], config=raglite_test_config)

    with Session(create_database_engine(raglite_test_config)) as session:
        chunk_ids = session.exec(
            select(Chunk.id).where(col(Chunk.document_id).in_([doc1_id]))
        ).all()

    deleted_count = delete_documents([doc1_id, "fake_id"], config=raglite_test_config)
    assert deleted_count == 1, f"Expected 1 document to be deleted, but got {deleted_count}"

    with Session(create_database_engine(raglite_test_config)) as session:
        # Check that the document is deleted
        assert session.exec(select(Document).where(Document.id == doc1_id)).first() is None, (
            "Document was not deleted"
        )
        # Check that tables with foreign keys to document have no entries for the deleted document
        for table_name, table in SQLModel.metadata.tables.items():
            document_fks = [fk for fk in table.foreign_keys if fk.column.table.name == "document"]
            for fk in document_fks:
                col_to_check = table.c[fk.parent.name]
                assert (
                    session.exec(select(col_to_check).where(col_to_check == doc1_id)).first()  # type: ignore[attr-defined]
                    is None
                ), f"{table_name} still contains data for deleted document"
        # Check that chunk embeddings are deleted
        assert (
            session.exec(
                select(ChunkEmbedding).where(col(ChunkEmbedding.chunk_id).in_(chunk_ids))
            ).first()
            is None
        ), "Chunk embeddings were not deleted"
        # Check that metadata fields are deleted
        existing_metadata = {
            record.name: record for record in _get_database_metadata(session=session)
        }
        assert "classification" not in existing_metadata, (
            "Metadata field 'classification' was not deleted"  # classification row should be deleted
        )
        assert "author" in existing_metadata, (
            "Author key should still exist for the Einstein fixture doc"
        )
        assert "Test Author" not in existing_metadata["author"].values, (
            "Metadata field 'author' was not deleted"  # row should remain with author: ['Albert Einstein']
        )


def test_delete_by_metadata(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion by metadata."""
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1, classification="A")
    insert_documents([document1], config=raglite_test_config)
    document2 = Document.from_text(content1 + " diff", classification="A")
    insert_documents([document2], config=raglite_test_config)

    deleted_count = delete_documents_by_metadata(
        {"classification": "A"}, config=raglite_test_config
    )
    assert deleted_count == 2, f"Expected 2 documents to be deleted, but got {deleted_count}"  # noqa: PLR2004
