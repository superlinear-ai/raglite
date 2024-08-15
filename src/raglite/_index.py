"""Index documents."""

from functools import partial
from pathlib import Path

import numpy as np
from pynndescent import NNDescent
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Document, VectorSearchChunkIndex, create_database_engine
from raglite._embed import embed_strings
from raglite._markdown import document_to_markdown
from raglite._split_chunks import split_chunks
from raglite._split_sentences import split_sentences
from raglite._typing import FloatMatrix


def _create_chunk_records(
    document_id: str,
    chunks: list[str],
    multi_vector_embeddings: list[FloatMatrix],
    config: RAGLiteConfig,
) -> list[Chunk]:
    """Process chunks into headings, body and improved multi-vector embeddings."""
    # Create the chunk records.
    chunk_records = []
    contextualized_chunks = []
    headings = ""
    for i, chunk in enumerate(chunks):
        # Create and append the contextualised chunk, which includes the current Markdown headings.
        contextualized_chunks.append(headings + "\n\n" + chunk)
        # Create and append the chunk record.
        chunk_record = Chunk.from_body(
            document_id=document_id, index=i, body=chunk, headings=headings
        )
        chunk_records.append(chunk_record)
        # Update the Markdown headings with those of this chunk.
        headings = chunk_record.extract_headings()
    # Embed the contextualised chunks.
    contextualized_embeddings = embed_strings(contextualized_chunks, config=config)
    # Update the chunk records with improved multi-vector embeddings that combine its multi-vector
    # embedding with its contextualised chunk embedding.
    for chunk_record, multi_vector_embedding, contextualized_embedding in zip(
        chunk_records, multi_vector_embeddings, contextualized_embeddings, strict=True
    ):
        chunk_embedding = (
            config.multi_vector_weight * multi_vector_embedding
            + (1 - config.multi_vector_weight) * contextualized_embedding[np.newaxis, :]
        )
        chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding, axis=1, keepdims=True)
        chunk_record.multi_vector_embedding = chunk_embedding
    return chunk_records


def insert_document(doc_path: Path, *, config: RAGLiteConfig | None = None) -> None:
    """Insert a document into the database."""
    # Use the default config if not provided.
    config = config or RAGLiteConfig()
    # Preprocess the document into chunks.
    with tqdm(total=4, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Initializing database")
        engine = create_database_engine(config.db_url)
        pbar.update(1)
        pbar.set_description("Converting to Markdown")
        doc = document_to_markdown(doc_path)
        pbar.update(1)
        pbar.set_description("Splitting sentences")
        sentences = split_sentences(doc, max_len=config.chunk_max_size)
        pbar.update(1)
        pbar.set_description("Splitting chunks")
        chunks, multi_vector_embeddings = split_chunks(
            sentences,
            max_size=config.chunk_max_size,
            sentence_window_size=config.chunk_sentence_window_size,
            embed=partial(embed_strings, config=config),
        )
        pbar.update(1)
    # Create and store the chunk records.
    with Session(engine) as session:
        # Add the document to the document table.
        document_record = Document.from_path(doc_path)
        if session.get(Document, document_record.id) is None:
            session.add(document_record)
            session.commit()
        # Create the chunk records.
        chunk_records = _create_chunk_records(
            document_record.id, chunks, multi_vector_embeddings, config
        )
        # Store the chunk records.
        for chunk_record in tqdm(
            chunk_records, desc="Storing chunks", unit="chunk", dynamic_ncols=True
        ):
            if session.get(Chunk, chunk_record.id) is not None:
                continue
            session.add(chunk_record)
            session.commit()


def update_vector_index(config: RAGLiteConfig | None = None) -> None:
    """Update the vector search chunk index with any unindexed chunks."""
    config = config or RAGLiteConfig()
    engine = create_database_engine(config.db_url)
    with Session(engine) as session:
        # Get the vector search chunk index from the database, or create a new one.
        vector_search_chunk_index = session.get(
            VectorSearchChunkIndex, config.vector_search_index_id
        ) or VectorSearchChunkIndex(id=config.vector_search_index_id)
        num_chunks_indexed = len(vector_search_chunk_index.chunk_sizes)
        # Get the unindexed chunks.
        statement = select(Chunk).offset(num_chunks_indexed)
        unindexed_chunks = session.exec(statement).all()
        num_chunks_unindexed = len(unindexed_chunks)
        # Index the unindexed chunks.
        with tqdm(
            total=num_chunks_indexed + num_chunks_unindexed,
            desc="Indexing chunks",
            unit="chunk",
            dynamic_ncols=True,
        ) as pbar:
            pbar.update(num_chunks_indexed)
            if num_chunks_unindexed == 0:
                return
            X_unindexed = np.vstack([chunk.multi_vector_embedding for chunk in unindexed_chunks])  # noqa: N806
            if num_chunks_indexed == 0:
                vector_search_chunk_index.index = NNDescent(
                    X_unindexed, metric=config.vector_search_index_metric
                )
                vector_search_chunk_index.index.prepare()
            else:
                vector_search_chunk_index.index.update(X_unindexed)  # type: ignore[union-attr]
                vector_search_chunk_index.index.prepare()  # type: ignore[union-attr]
            vector_search_chunk_index.chunk_sizes.extend(
                [chunk.multi_vector_embedding.shape[0] for chunk in unindexed_chunks]
            )
            # Store the updated vector search chunk index.
            session.add(vector_search_chunk_index)
            session.commit()
            pbar.update(num_chunks_unindexed)
