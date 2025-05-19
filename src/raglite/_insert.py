"""Index documents."""

from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Document, create_database_engine
from raglite._embed import embed_strings, embed_strings_without_late_chunking, embedding_type
from raglite._markdown import document_to_markdown
from raglite._split_chunklets import split_chunklets
from raglite._split_chunks import split_chunks
from raglite._split_sentences import split_sentences
from raglite._typing import FloatMatrix


def _create_chunk_records(
    document: Document,
    chunks: list[str],
    chunk_embeddings: list[FloatMatrix],
    metadata: dict[str, str],
    config: RAGLiteConfig,
) -> tuple[list[Chunk], list[list[ChunkEmbedding]]]:
    """Process chunks into chunk and chunk embedding records."""
    # Create the chunk records.
    chunk_records, headings = [], ""
    for i, chunk in enumerate(chunks):
        # Create and append the chunk record.
        record = Chunk.from_body(
            document=document, index=i, body=chunk, headings=headings, **metadata
        )
        chunk_records.append(record)
        # Update the Markdown headings with those of this chunk.
        headings = record.extract_headings()
    # Create the chunk embedding records.
    chunk_embedding_records = []
    if embedding_type(config=config) == "late_chunking":
        # Every chunk record is associated with a list of chunk embedding records, one for each of
        # the chunklets in the chunk.
        for chunk_record, chunk_embedding in zip(chunk_records, chunk_embeddings, strict=True):
            chunk_embedding_records.append(
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
                chunk_embedding_records.append(
                    [
                        ChunkEmbedding(
                            chunk_id=chunk_record.id,
                            embedding=α * chunklet_embedding + (1 - α) * full_chunk_embedding,
                        )
                        for chunklet_embedding in chunk_embedding
                    ]
                )
            else:
                chunk_embedding_records.append(
                    [
                        ChunkEmbedding(
                            chunk_id=chunk_record.id,
                            embedding=full_chunk_embedding,
                        )
                    ]
                )
    return chunk_records, chunk_embedding_records


def insert_document(
    source: Path | str,
    *,
    filename: str | None = None,
    metadata: dict[str, str] | None = None,
    config: RAGLiteConfig | None = None,
) -> None:
    """Insert a document into the database and update the index.

    Parameters
    ----------
    source
        A document file path or the document's content as a Markdown string.
    filename
        The document filename to use if the source is a Markdown string. If not provided, the first
        line of the source is used.
    metadata
        Document metadata that is attached to the extracted chunks.
    config
        The RAGLite config to use to insert the document into the database.

    Returns
    -------
        None
    """
    # Use the default config if not provided.
    config = config or RAGLiteConfig()
    # Preprocess the document into chunks and chunk embeddings.
    total_steps = 7
    with tqdm(total=total_steps, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Initializing database")
        engine = create_database_engine(config)
        pbar.update(1)
        # Create document record based on input type.
        pbar.set_description("Converting to Markdown")
        document_record, doc = (
            (Document.from_path(source), document_to_markdown(source))
            if isinstance(source, Path)
            else (Document.from_markdown(source, filename=filename), source)
        )
        with Session(engine) as session:  # Exit early if the document is already in the database.
            if session.get(Document, document_record.id) is not None:
                pbar.set_description("Document already in database")
                pbar.update(total_steps - 1)
                pbar.close()
                return
        pbar.update(1)
        pbar.set_description("Splitting sentences")
        sentences = split_sentences(doc, max_len=config.chunk_max_size)
        pbar.update(1)
        pbar.set_description("Splitting chunklets")
        chunklets = split_chunklets(sentences, max_size=config.chunk_max_size)
        pbar.update(1)
        pbar.set_description("Embedding chunklets")
        chunklet_embeddings = embed_strings(chunklets, config=config)
        pbar.update(1)
        pbar.set_description("Splitting chunks")
        chunks, chunk_embeddings = split_chunks(
            chunklets=chunklets,
            chunklet_embeddings=chunklet_embeddings,
            max_size=config.chunk_max_size,
        )
        pbar.update(1)
        pbar.set_description("Updating database")
        with Session(engine) as session:
            session.add(document_record)
            for chunk_record, chunk_embedding_record_list in zip(
                *_create_chunk_records(
                    document_record, chunks, chunk_embeddings, metadata or {}, config
                ),
                strict=True,
            ):
                session.add(chunk_record)
                session.add_all(chunk_embedding_record_list)
            session.commit()
            if engine.dialect.name == "duckdb":
                # DuckDB does not automatically update its keyword search index [1], so we do it
                # manually after insertion.
                # [1] https://duckdb.org/docs/stable/extensions/full_text_search
                session.execute(
                    text("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1);")
                )
                session.commit()
        pbar.update(1)
        pbar.close()
