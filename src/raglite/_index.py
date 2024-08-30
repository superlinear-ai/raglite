"""Index documents."""

from functools import partial
from pathlib import Path

import numpy as np
from sqlalchemy.engine import make_url
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Document, IndexMetadata, create_database_engine
from raglite._embed import embed_strings
from raglite._markdown import document_to_markdown
from raglite._split_chunks import split_chunks
from raglite._split_sentences import split_sentences
from raglite._typing import FloatMatrix


def _create_chunk_records(
    document_id: str,
    chunks: list[str],
    sentence_embeddings: list[FloatMatrix],
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
    # Embed the contextualised chunks, which include the current Markdown headings.
    contextualized_embeddings = embed_strings([str(chunk) for chunk in chunks], config=config)
    # Set the chunk's multi-vector embedding as a linear combination of its sentence embeddings
    # (for local context) and an embedding of the contextualised chunk (for global context).
    α = config.sentence_embedding_weight  # noqa: PLC2401
    chunk_embedding_records = []
    for chunk_record, sentence_embedding, contextualized_embedding in zip(
        chunk_records, sentence_embeddings, contextualized_embeddings, strict=True
    ):
        chunk_embedding = α * sentence_embedding + (1 - α) * contextualized_embedding[np.newaxis, :]
        chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding, axis=1, keepdims=True)
        chunk_embedding_records.append(
            [ChunkEmbedding(chunk_id=chunk_record.id, embedding=row) for row in chunk_embedding]
        )
    return chunk_records, chunk_embedding_records


def insert_document(doc_path: Path, *, config: RAGLiteConfig | None = None) -> None:
    """Insert a document into the database and update the index."""
    # Use the default config if not provided.
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # Preprocess the document into chunks.
    with tqdm(total=4, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Initializing database")
        engine = create_database_engine(config)
        pbar.update(1)
        pbar.set_description("Converting to Markdown")
        doc = document_to_markdown(doc_path)
        pbar.update(1)
        pbar.set_description("Splitting sentences")
        sentences = split_sentences(doc, max_len=config.chunk_max_size)
        pbar.update(1)
        pbar.set_description("Splitting chunks")
        chunks, sentence_embeddings = split_chunks(
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
        # Create the chunk records to insert into the chunk table.
        chunk_records, chunk_embedding_records = _create_chunk_records(
            document_record.id, chunks, sentence_embeddings, config
        )
        # Store the chunk and chunk embedding records.
        for chunk_record, chunk_embedding_record in tqdm(
            zip(chunk_records, chunk_embedding_records, strict=True),
            desc="Storing chunks" if db_backend == "sqlite" else "Storing and indexing chunks",
            total=len(chunk_records),
            unit="chunk",
            dynamic_ncols=True,
        ):
            if session.get(Chunk, chunk_record.id) is not None:
                continue
            session.add(chunk_record)
            session.add_all(chunk_embedding_record)
            session.commit()
    # Manually update the vector search chunk index for SQLite.
    if db_backend == "sqlite":
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
