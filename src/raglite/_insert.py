"""Index documents."""

from pathlib import Path

import mdformat
import numpy as np
from sqlalchemy.engine import make_url
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Document, IndexMetadata, create_database_engine
from raglite._embed import embed_sentences, sentence_embedding_type
from raglite._markdown import document_to_markdown
from raglite._split_chunks import split_chunks
from raglite._split_sentences import split_sentences
from raglite._typing import DocumentId, FloatMatrix


def _create_chunk_records(
    document_id: DocumentId,
    chunks: list[str],
    chunk_embeddings: list[FloatMatrix],
    config: RAGLiteConfig,
) -> tuple[list[Chunk], list[list[ChunkEmbedding]]]:
    """Process chunks into chunk and chunk embedding records."""
    # Create the chunk records.
    chunk_records, headings = [], ""
    for i, chunk in enumerate(chunks):
        # Create and append the chunk record.
        record = Chunk.from_body(document_id=document_id, index=i, body=chunk, headings=headings)
        chunk_records.append(record)
        # Update the Markdown headings with those of this chunk.
        headings = record.extract_headings()
    # Create the chunk embedding records.
    chunk_embedding_records = []
    if sentence_embedding_type(config=config) == "late_chunking":
        # Every chunk record is associated with a list of chunk embedding records, one for each of
        # the sentences in the chunk.
        for chunk_record, chunk_embedding in zip(chunk_records, chunk_embeddings, strict=True):
            chunk_embedding_records.append(
                [
                    ChunkEmbedding(chunk_id=chunk_record.id, embedding=sentence_embedding)
                    for sentence_embedding in chunk_embedding
                ]
            )
    else:
        # Embed the full chunks, including the current Markdown headings.
        full_chunk_embeddings = embed_sentences([str(chunk) for chunk in chunks], config=config)
        # Every chunk record is associated with a list of chunk embedding records. The chunk
        # embedding records each correspond to a linear combination of a sentence embedding and an
        # embedding of the full chunk with Markdown headings.
        α = 0.382  # Golden ratio.  # noqa: PLC2401
        for chunk_record, chunk_embedding, full_chunk_embedding in zip(
            chunk_records, chunk_embeddings, full_chunk_embeddings, strict=True
        ):
            chunk_embedding_records.append(
                [
                    ChunkEmbedding(
                        chunk_id=chunk_record.id,
                        embedding=α * sentence_embedding + (1 - α) * full_chunk_embedding,
                    )
                    for sentence_embedding in chunk_embedding
                ]
            )
    return chunk_records, chunk_embedding_records


def insert_document(  # noqa: PLR0915
    source: Path | str,
    *,
    filename: str | None = None,
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
    config
        The RAGLite config to use to insert the document into the database.

    Returns
    -------
        None
    """
    # Use the default config if not provided.
    config = config or RAGLiteConfig()

    # Preprocess the document into chunks and chunk embeddings.
    with tqdm(total=6, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Initializing database")
        engine = create_database_engine(config)
        pbar.update(1)
        # Create document record based on input type.
        pbar.set_description("Converting and formatting Markdown")
        document_record, doc = (
            (Document.from_path(source), document_to_markdown(source))
            if isinstance(source, Path)
            else (Document.from_markdown(source, filename=filename), mdformat.text(source))
        )
        with Session(engine) as session:  # Exit early if the document is already in the database.
            if session.get(Document, document_record.id) is not None:
                pbar.set_description("Document already in database")
                pbar.update(5)
                pbar.close()
                return
        pbar.update(1)
        pbar.set_description("Splitting sentences")
        sentences = split_sentences(doc, max_len=config.chunk_max_size)
        pbar.update(1)
        pbar.set_description("Embedding sentences")
        sentence_embeddings = embed_sentences(sentences, config=config)
        pbar.update(1)
        pbar.set_description("Splitting chunks")
        chunks, chunk_embeddings = split_chunks(
            sentences=sentences,
            sentence_embeddings=sentence_embeddings,
            sentence_window_size=config.embedder_sentence_window_size,
            max_size=config.chunk_max_size,
        )
        pbar.update(1)
        pbar.set_description("Updating database")
        with Session(engine) as session:
            session.add(document_record)
            for chunk_record, chunk_embedding_record_list in zip(
                *_create_chunk_records(document_record.id, chunks, chunk_embeddings, config),
                strict=True,
            ):
                session.add(chunk_record)
                session.add_all(chunk_embedding_record_list)
            session.commit()
        pbar.update(1)
        pbar.close()
    # Manually update the vector search chunk index for SQLite.
    if make_url(config.db_url).get_backend_name() == "sqlite":
        from pynndescent import NNDescent

        with Session(engine) as session:
            # Get the vector search chunk index from the database, or create a new one.
            index_metadata = session.get(IndexMetadata, "default") or IndexMetadata(id="default")
            chunk_ids = index_metadata.metadata_.get("chunk_ids", [])
            chunk_sizes = index_metadata.metadata_.get("chunk_sizes", [])
            # Get the unindexed chunks.
            unindexed_chunks = list(session.exec(select(Chunk).offset(len(chunk_ids))).all())
            if not unindexed_chunks:
                return
            # Assemble the unindexed chunk embeddings into a NumPy array.
            unindexed_chunk_embeddings = [chunk.embedding_matrix for chunk in unindexed_chunks]
            X = np.vstack(unindexed_chunk_embeddings)  # noqa: N806
            # Index the unindexed chunks.
            with tqdm(
                total=len(unindexed_chunks),
                desc="Indexing chunks",
                unit="chunk",
                dynamic_ncols=True,
            ) as pbar:
                # Fit or update the ANN index.
                if len(chunk_ids) == 0:
                    nndescent = NNDescent(X, metric=config.vector_search_index_metric)
                else:
                    nndescent = index_metadata.metadata_["index"]
                    nndescent.update(X)
                # Prepare the ANN index so it can to handle query vectors not in the training set.
                nndescent.prepare()
                # Update the index metadata and mark it as dirty by recreating the dictionary.
                index_metadata.metadata_ = {
                    **index_metadata.metadata_,
                    "index": nndescent,
                    "chunk_ids": chunk_ids + [c.id for c in unindexed_chunks],
                    "chunk_sizes": chunk_sizes + [len(em) for em in unindexed_chunk_embeddings],
                }
                # Store the updated vector search chunk index.
                session.add(index_metadata)
                session.commit()
                pbar.update(len(unindexed_chunks))
