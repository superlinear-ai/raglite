"""Index documents."""

from pathlib import Path

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


def insert_document(doc_path: Path, *, config: RAGLiteConfig | None = None) -> None:  # noqa: PLR0915
    """Insert a document into the database and update the index."""
    # Use the default config if not provided.
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # Preprocess the document into chunks and chunk embeddings.
    with tqdm(total=5, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Initializing database")
        engine = create_database_engine(config)
        pbar.update(1)
        pbar.set_description("Converting to Markdown")
        doc = document_to_markdown(doc_path)
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
    # Create and store the chunk records.
    with Session(engine) as session:
        # Add the document to the document table.
        document_record = Document.from_path(doc_path)
        if session.get(Document, document_record.id) is None:
            session.add(document_record)
            session.commit()
        # Create the chunk records to insert into the chunk table.
        chunk_records, chunk_embedding_records = _create_chunk_records(
            document_record.id, chunks, chunk_embeddings, config
        )
        # Store the chunk and chunk embedding records.
        for chunk_record, chunk_embedding_record_list in tqdm(
            zip(chunk_records, chunk_embedding_records, strict=True),
            desc="Inserting chunks",
            total=len(chunk_records),
            unit="chunk",
            dynamic_ncols=True,
        ):
            if session.get(Chunk, chunk_record.id) is not None:
                continue
            session.add(chunk_record)
            session.add_all(chunk_embedding_record_list)
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
