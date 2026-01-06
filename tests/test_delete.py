"""Test RAGLite's document deletion."""

from sqlmodel import Session, SQLModel, select

from raglite._config import RAGLiteConfig
from raglite._database import (
    Chunk,
    ChunkEmbedding,
    Document,
    create_database_engine,
)
from raglite._delete import delete_documents, delete_documents_by_metadata
from raglite._insert import insert_documents


def test_delete(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion."""
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1)
    doc1_id = document1.id
    insert_documents([document1], config=raglite_test_config)

    with Session(create_database_engine(raglite_test_config)) as session:
        chunks_ids = session.exec(select(Chunk.id).where(Chunk.document_id == doc1_id)).all()
        assert chunks_ids is not None, "Chunks were not found"
        chunk_embeddings_ids = session.exec(
            select(ChunkEmbedding.id).where(ChunkEmbedding.chunk_id.in_(chunks_ids))  # type: ignore[attr-defined]
        ).all()
        assert chunk_embeddings_ids is not None, "Chunk embeddings were not found"

    deleted_count = delete_documents([doc1_id], config=raglite_test_config)
    assert deleted_count == 1, f"Expected 1 document to be deleted, but got {deleted_count}"

    with Session(create_database_engine(raglite_test_config)) as session:
        assert session.exec(select(Document).where(Document.id == doc1_id)).first() is None, (
            "Document was not deleted"
        )
        # Check that other tables are deleted
        for table in SQLModel.metadata.tables.values():
            if "document_id" in table.c:
                assert (
                    session.exec(select(table).where(table.c.document_id == doc1_id)).first()  # type: ignore[call-overload]
                    is None
                ), f"{table.name} was not deleted"
        assert (
            session.exec(
                select(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunks_ids))  # type: ignore[attr-defined]
            ).first()
            is None
        ), "Chunk embeddings were not deleted"


def test_delete_by_metadata(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion by metadata."""
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1, metadata="A")
    document2 = Document.from_text(content1 + " diff", metadata="A")
    insert_documents([document1, document2], config=raglite_test_config)

    deleted_count = delete_documents_by_metadata({"metadata": "A"}, config=raglite_test_config)
    assert deleted_count == 2, f"Expected 2 documents to be deleted, but got {deleted_count}"  # noqa: PLR2004
