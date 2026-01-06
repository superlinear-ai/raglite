"""Delete documents from the database."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from filelock import FileLock
from sqlalchemy import delete, text
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Session, col, select

from raglite._config import RAGLiteConfig
from raglite._database import (
    Chunk,
    ChunkEmbedding,
    Document,
    Eval,
    IndexMetadata,
    _adapt_metadata,
    create_database_engine,
)
from raglite._typing import DocumentId


def _rebuild_indexes(session: Session, engine: Engine) -> None:
    """Rebuild search indexes after deletion (DuckDB only).

    Parameters
    ----------
    session
        Active database session.
    engine
        Database engine.
    """
    if engine.dialect.name == "duckdb":
        # DuckDB does not automatically update its keyword search index, so we rebuild it
        # manually after deletion. Additionally, we re-compact the HNSW index. Finally, we
        # synchronize data in the write-ahead log (WAL) to the database data file with the
        # CHECKPOINT statement.
        session.execute(text("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1);"))
        session.execute(text("PRAGMA hnsw_compact_index('vector_search_chunk_index');"))
        session.commit()
        session.execute(text("CHECKPOINT;"))
    # PostgreSQL indexes update automatically, so no action needed


def _invalidate_query_adapter(session: Session) -> None:
    """Remove cached query adapter from IndexMetadata.

    Parameters
    ----------
    session
        Active database session.
    """
    # Delete the IndexMetadata record with id="default" if it exists
    index_metadata = session.get(IndexMetadata, "default")
    if index_metadata is not None:
        session.delete(index_metadata)

    # Clear the LRU cache for IndexMetadata._get()
    IndexMetadata._get.cache_clear()  # noqa: SLF001


def delete_documents(
    document_ids: list[DocumentId],
    *,
    config: RAGLiteConfig | None = None,
) -> int:
    """Delete documents from the database and update the index.

    Important Notes
    ---------------
    - **PostgreSQL**: Deletion is atomic. All documents and related data are deleted
      in a single transaction.
    - **DuckDB**: Due to DuckDB's immediate foreign key constraint checking, deletion
      uses multiple commits and is NOT atomic. If an error occurs mid-deletion, some
      data may be deleted while other data remains. This is a known DuckDB limitation.

    Parameters
    ----------
    document_ids
        A list of document IDs to delete from the database.
    config
        The RAGLite config to use to delete the documents from the database.

    Returns
    -------
    int
        The number of documents deleted.
    """
    # Early exit if no document IDs are provided
    if not document_ids:
        return 0

    # Create database engine and session
    engine = create_database_engine(config := config or RAGLiteConfig())

    # For DuckDB databases, acquire a lock on the database
    if engine.dialect.name == "duckdb":
        db_url = make_url(config.db_url) if isinstance(config.db_url, str) else config.db_url
        db_lock = (
            FileLock(Path(db_url.database).with_suffix(".lock"))
            if db_url.database
            else nullcontext()
        )
    else:
        db_lock = nullcontext()

    # Delete documents and perform cleanup
    with db_lock, Session(engine) as session:
        if engine.dialect.name == "postgresql":
            # PostgreSQL: Use ORM cascade delete for atomic transactions
            # PostgreSQL supports deferred constraint checking, so this works atomically
            deleted_count = 0
            for document_id in document_ids:
                document = session.get(Document, document_id)
                if document is not None:
                    session.delete(document)  # Cascade handles children
                    deleted_count += 1

            _invalidate_query_adapter(session)
            session.commit()
        else:
            # DuckDB: Use manual cascade with intermediate commits
            # DuckDB checks FK constraints immediately, so we must commit after each step
            # This is a known issue: https://github.com/duckdb/duckdb/issues/13819
            # Limitations: https://duckdb.org/docs/stable/sql/indexes#over-eager-constraint-checking-in-foreign-keys
            # WARNING: This is NOT atomic - failures may leave partial deletions

            # Find all chunks for the documents to be deleted
            chunk_ids = session.exec(
                select(Chunk.id).where(col(Chunk.document_id).in_(document_ids))
            ).all()

            # Delete chunk embeddings (deepest dependency)
            if chunk_ids:
                session.execute(
                    delete(ChunkEmbedding).where(col(ChunkEmbedding.chunk_id).in_(chunk_ids))
                )
                session.commit()

            # Delete chunks
            if chunk_ids:
                session.execute(delete(Chunk).where(col(Chunk.id).in_(chunk_ids)))
                session.commit()

            # Delete evals
            session.execute(delete(Eval).where(col(Eval.document_id).in_(document_ids)))
            session.commit()

            # Delete documents and count
            result = session.execute(
                delete(Document)
                .where(col(Document.id).in_(document_ids))
                .returning(col(Document.id))
            )
            deleted_count = len(result.all())
            session.commit()

            # Invalidate query adapter cache
            _invalidate_query_adapter(session)

            # Rebuild indexes (DuckDB only)
            _rebuild_indexes(session, engine)

            # Final commit
            session.commit()

    return deleted_count


def delete_documents_by_metadata(
    metadata_filter: dict[str, Any],
    *,
    config: RAGLiteConfig | None = None,
) -> int:
    """Delete documents matching a metadata filter from the database and update the index.

    Parameters
    ----------
    metadata_filter
        A dictionary of metadata key-value pairs to match. All key-value pairs must match
        for a document to be deleted.
    config
        The RAGLite config to use to delete the documents from the database.

    Returns
    -------
    int
        The number of documents deleted.

    Raises
    ------
    ValueError
        If metadata_filter is empty.
    """
    # Validate that metadata_filter is not empty
    if not metadata_filter:
        error_message = (
            "metadata_filter cannot be empty to prevent accidental deletion of all documents"
        )
        raise ValueError(error_message)

    # Create database engine and session to query matching documents
    engine = create_database_engine(config := config or RAGLiteConfig())

    # Normalize metadata filter values to lists to match stored metadata
    metadata_filter = _adapt_metadata(metadata_filter)

    with Session(engine) as session:
        # Query all documents and filter in Python to match all metadata key-value pairs
        # We filter in Python because metadata is stored as JSON and complex JSON queries
        # are not portable across DuckDB and PostgreSQL
        all_documents = session.exec(select(Document)).all()

        # Filter documents where all metadata key-value pairs match
        # Note: User-provided metadata may be nested under a 'metadata' key
        document_ids = []
        for doc in all_documents:
            # Check both top-level and nested metadata
            metadata_to_check = doc.metadata_
            if "metadata" in doc.metadata_ and isinstance(doc.metadata_["metadata"], dict):
                metadata_to_check = doc.metadata_["metadata"]

            # Check if all filter key-value pairs match
            if all(
                key in metadata_to_check and metadata_to_check[key] == value
                for key, value in metadata_filter.items()
            ):
                document_ids.append(doc.id)

    # Use delete_documents to perform the actual deletion
    return delete_documents(document_ids, config=config)
