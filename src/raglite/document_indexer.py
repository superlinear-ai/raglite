"""Index documents."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from pynndescent import NNDescent
from sqlalchemy.engine import URL
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite.chunk_splitter import split_chunks
from raglite.database_models import Chunk, ChunkANNIndex, Document, create_database_engine
from raglite.markdown_converter import document_to_markdown
from raglite.sentence_splitter import split_sentences
from raglite.string_embedder import embed_strings


def create_chunk_records(
    document_id: str,
    chunks: list[str],
    multi_vector_embeddings: list[np.ndarray],
    multi_vector_weight: float = 0.2,
    embed: Callable[[list[str]], np.ndarray] = embed_strings,
) -> list[Chunk]:
    """Process chunks into headers, body and improved embeddings."""
    # Create the chunk records.
    chunk_records = []
    contextualized_chunks = []
    headers = ""
    for i, chunk in enumerate(chunks):
        # Create and append the contextualised chunk, which includes the current Markdown headers.
        contextualized_chunks.append(headers + "\n\n" + chunk)
        # Create and append the chunk record.
        chunk_record = Chunk.from_body(
            document_id=document_id, index=i, body=chunk, headers=headers
        )
        chunk_records.append(chunk_record)
        # Update the Markdown headers with those of this chunk.
        headers = chunk_record.extract_headers()
    # Embed the contextualised chunks.
    contextualized_embeddings = embed(contextualized_chunks)
    contextualized_embeddings = contextualized_embeddings / np.linalg.norm(
        contextualized_embeddings, axis=1, keepdims=True
    )
    # Update the chunk records with improved multi vector embeddings that combine its multi vector
    # embedding with its contextualised chunk embedding.
    for chunk_record, multi_vector_embedding, contextualized_embedding in zip(
        chunk_records, multi_vector_embeddings, contextualized_embeddings, strict=True
    ):
        chunk_embedding = (
            multi_vector_weight * multi_vector_embedding
            + (1 - multi_vector_weight) * contextualized_embedding[np.newaxis, :]
        )
        chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding, axis=1, keepdims=True)
        chunk_record.multi_vector_embedding = chunk_embedding
    return chunk_records


def insert_document(doc_path: Path, db_url: str | URL = "sqlite:///raglite.sqlite") -> None:
    """Insert a document."""
    # Preprocess the document into chunks.
    with tqdm(total=4, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Initializing database")
        engine = create_database_engine(db_url)
        pbar.update(1)
        pbar.set_description("Converting to Markdown")
        doc = document_to_markdown(doc_path)
        pbar.update(1)
        pbar.set_description("Splitting sentences")
        sentences = split_sentences(doc)
        pbar.update(1)
        pbar.set_description("Splitting chunks")
        chunks, multi_vector_embeddings = split_chunks(sentences)
        pbar.update(1)
    # Create and store the chunk records.
    with Session(engine) as session:
        # Add the document to the document table.
        document_record = Document.from_path(doc_path)
        if session.get(Document, document_record.id) is None:
            session.add(document_record)
            session.commit()
        # Create the chunk records.
        chunk_records = create_chunk_records(document_record.id, chunks, multi_vector_embeddings)
        # Store the chunk records.
        for chunk_record in tqdm(
            chunk_records, desc="Storing chunks", unit="chunk", dynamic_ncols=True
        ):
            if session.get(Chunk, chunk_record.id) is not None:
                continue
            session.add(chunk_record)
            session.commit()


def update_index(index_id: str = "default", db_url: str | URL = "sqlite:///raglite.sqlite") -> None:
    """Update the chunk ANN index with any unindexed chunks."""
    engine = create_database_engine(db_url)
    with Session(engine) as session:
        # Get the chunk ANN index from the database, or create a new one.
        chunk_ann_index = session.get(ChunkANNIndex, index_id) or ChunkANNIndex(id=index_id)
        num_chunks_indexed = len(chunk_ann_index.chunk_sizes)
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
                chunk_ann_index.index = NNDescent(X_unindexed, metric="cosine", compressed=True)
            else:
                chunk_ann_index.index.prepare()
                chunk_ann_index.index.update(X_unindexed)
            chunk_ann_index.chunk_sizes.extend(
                [chunk.multi_vector_embedding.shape[0] for chunk in unindexed_chunks]
            )
            pbar.update(num_chunks_unindexed)
            # Store the updated chunk ANN index.
            session.add(chunk_ann_index)
            session.commit()
