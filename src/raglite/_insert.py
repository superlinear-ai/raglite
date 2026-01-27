"""Index documents."""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from filelock import FileLock
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import (
    Chunk,
    ChunkEmbedding,
    Document,
    Metadata,
    create_database_engine,
)
from raglite._embed import embed_strings, embed_strings_without_late_chunking, embedding_type
from raglite._split_chunklets import split_chunklets
from raglite._split_chunks import split_chunks
from raglite._split_sentences import split_sentences
from raglite._typing import MetadataValue

METADATA_EXCLUDED_FIELDS = ["filename", "uri", "url", "size", "created", "modified"]


def _get_database_metadata(
    session: Session | None = None,
    config: RAGLiteConfig | None = None,
) -> Sequence[Metadata]:
    """Fetch all metadata records from the database."""
    if session:
        return session.exec(select(Metadata)).all()
    with Session(create_database_engine(config or RAGLiteConfig())) as new_session:
        return new_session.exec(select(Metadata)).all()


def _aggregate_metadata_from_documents(
    documents: list[Document],
    metadata_excluded_fields: list[str] = METADATA_EXCLUDED_FIELDS,
) -> dict[str, set[MetadataValue]]:
    """Aggregate metadata values from all documents."""
    metadata: dict[str, set[MetadataValue]] = {}
    for doc in documents:
        for key, value in doc.metadata_.items():
            if key in metadata_excluded_fields:
                continue
            if key not in metadata:
                metadata[key] = set()
            if isinstance(value, list):
                metadata[key].update(value)
            else:
                metadata[key].add(value)
    return metadata


def _update_metadata_from_documents(
    session: Session,
    documents: list[Document],
) -> None:
    """Update or add metadata records."""
    if not documents:
        return
    metadata = _aggregate_metadata_from_documents(documents=documents)
    existing_metadata = {record.name: record for record in _get_database_metadata(session=session)}
    # Update or add metadata records.
    for key, values in metadata.items():
        # Update
        if key in existing_metadata:
            result = existing_metadata[key]
            values_to_add = set(values) - set(result.values)
            if values_to_add:
                result.values.extend(values_to_add)
                flag_modified(result, "values")  # Notify SQLAlchemy of the change
                session.add(result)
        # Add
        else:
            session.add(Metadata(name=key, values=list(values)))


def _create_chunk_records(
    document: Document, config: RAGLiteConfig
) -> tuple[Document, list[Chunk], list[list[ChunkEmbedding]]]:
    """Process chunks into chunk and chunk embedding records."""
    # Partition the document into chunks.
    assert document.content is not None
    sentences = split_sentences(document.content, max_len=config.chunk_max_size)
    chunklets = split_chunklets(sentences, max_size=config.chunk_max_size)
    chunklet_embeddings = embed_strings(chunklets, config=config)
    chunks, chunk_embeddings = split_chunks(
        chunklets=chunklets,
        chunklet_embeddings=chunklet_embeddings,
        max_size=config.chunk_max_size,
    )
    # Create the chunk records.
    chunk_records, headings = [], ""
    for i, chunk in enumerate(chunks):
        # Create and append the chunk record.
        record = Chunk.from_body(
            document=document, index=i, body=chunk, headings=headings, **document.metadata_
        )
        chunk_records.append(record)
        # Update the Markdown headings with those of this chunk.
        headings = record.extract_headings()
    # Create the chunk embedding records.
    chunk_embedding_records_list = []
    if embedding_type(config=config) == "late_chunking":
        # Every chunk record is associated with a list of chunk embedding records, one for each of
        # the chunklets in the chunk.
        for chunk_record, chunk_embedding in zip(chunk_records, chunk_embeddings, strict=True):
            chunk_embedding_records_list.append(
                [
                    ChunkEmbedding(chunk_id=chunk_record.id, embedding=chunklet_embedding)
                    for chunklet_embedding in chunk_embedding
                ]
            )
    else:
        # Embed the full chunks, including the current Markdown headings.
        full_chunk_embeddings = embed_strings_without_late_chunking(
            [chunk_record.content for chunk_record in chunk_records], config=config
        )
        # Every chunk record is associated with a list of chunk embedding records. The chunk
        # embedding records each correspond to a linear combination of a chunklet embedding and an
        # embedding of the full chunk with Markdown headings.
        α = 0.15  # Benchmark-optimised value.  # noqa: PLC2401
        for chunk_record, chunk_embedding, full_chunk_embedding in zip(
            chunk_records, chunk_embeddings, full_chunk_embeddings, strict=True
        ):
            if config.vector_search_multivector:
                chunk_embedding_records_list.append(
                    [
                        ChunkEmbedding(
                            chunk_id=chunk_record.id,
                            embedding=α * chunklet_embedding + (1 - α) * full_chunk_embedding,
                        )
                        for chunklet_embedding in chunk_embedding
                    ]
                )
            else:
                chunk_embedding_records_list.append(
                    [
                        ChunkEmbedding(
                            chunk_id=chunk_record.id,
                            embedding=full_chunk_embedding,
                        )
                    ]
                )
    return document, chunk_records, chunk_embedding_records_list


# Cap the number of worker threads to avoid excessive resource usage.
MAX_DEFAULT_WORKERS = 4  # Prevents oversubscription and high memory usage.


def insert_documents(  # noqa: C901, PLR0912
    documents: list[Document],
    *,
    max_workers: int | None = None,
    config: RAGLiteConfig | None = None,
) -> None:
    """Insert documents into the database and update the index.

    Parameters
    ----------
    documents
        A list of documents to insert into the database.
    max_workers
        The maximum number of worker threads to use.
    config
        The RAGLite config to use to insert the documents into the database.

    Returns
    -------
        None
    """
    # Verify that all documents have content.
    if not all(isinstance(doc.content, str) for doc in documents):
        error_message = "Some or all documents have missing `document.content`."
        raise ValueError(error_message)

    # Early exit if no documents are provided.
    # Remove duplicate documents and empty documents.
    documents = list({doc.id: doc for doc in documents}.values())
    documents = [doc for doc in documents if doc.content.strip()]  # type: ignore[union-attr]
    if not documents:
        return

    # Skip documents that are already in the database.
    batch_size = 128
    with Session(engine := create_database_engine(config := config or RAGLiteConfig())) as session:
        existing_doc_ids: set[str] = set()
        for i in range(0, len(documents), batch_size):
            doc_id_batch = [doc.id for doc in documents[i : i + batch_size]]
            existing_doc_ids.update(
                session.exec(select(Document.id).where(col(Document.id).in_(doc_id_batch))).all()
            )
        documents = [doc for doc in documents if doc.id not in existing_doc_ids]
        if not documents:
            return

    # Heuristic based on cpu count, amount of documents, and a cap of 4.
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(documents), MAX_DEFAULT_WORKERS)

    # For DuckDB databases, acquire a lock on the database.
    if engine.dialect.name == "duckdb":
        db_url = make_url(config.db_url) if isinstance(config.db_url, str) else config.db_url
        db_lock = (
            FileLock(Path(db_url.database).with_suffix(".lock"))
            if db_url.database
            else nullcontext()
        )
    else:
        db_lock = nullcontext()
    # Create and insert the document, chunk, and chunk embedding records.
    with (
        db_lock,
        Session(engine, expire_on_commit=False) as session,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(
            total=len(documents),
            desc="Inserting documents",
            unit="document",
            dynamic_ncols=True,
            leave=False,
        ) as pbar,
    ):
        futures = [
            executor.submit(partial(_create_chunk_records, config=config), doc) for doc in documents
        ]
        num_unflushed_embeddings = 0
        for future in as_completed(futures):
            try:
                document_record, chunk_records, chunk_embedding_records_list = future.result()
            except Exception as e:
                executor.shutdown(cancel_futures=True)  # Cancel remaining work.
                session.rollback()  # Cancel uncommitted changes.
                error_message = f"Error processing document: {e}"
                raise ValueError(error_message) from e
            session.add(document_record)
            session.add_all(chunk_records)
            for chunk_embedding_records in chunk_embedding_records_list:
                session.add_all(chunk_embedding_records)
                num_unflushed_embeddings += len(chunk_embedding_records)
            if num_unflushed_embeddings >= batch_size:
                session.flush()  # Flush changes to the database.
                session.expunge_all()  # Release memory of flushed changes.
                num_unflushed_embeddings = 0
            pbar.update()
        # Update metadata table.
        _update_metadata_from_documents(documents=documents, session=session)
        session.commit()
        if engine.dialect.name == "duckdb":
            # DuckDB does not automatically update its keyword search index [1], so we do it
            # manually after insertion. Additionally, we re-compact the HNSW index [2]. Finally, we
            # synchronize data in the write-ahead log (WAL) to the database data file with the
            # CHECKPOINT statement [3].
            # [1] https://duckdb.org/docs/stable/extensions/full_text_search
            # [2] https://duckdb.org/docs/stable/core_extensions/vss.html#inserts-updates-deletes-and-re-compaction
            # [3] https://duckdb.org/docs/stable/sql/statements/checkpoint.html
            session.execute(text("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1);"))
            if len(documents) >= 8:  # noqa: PLR2004
                session.execute(text("PRAGMA hnsw_compact_index('vector_search_chunk_index');"))
            session.commit()
            session.execute(text("CHECKPOINT;"))
