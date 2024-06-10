"""Index documents."""

from pathlib import Path

import numpy as np
from pynndescent import NNDescent
from sqlalchemy.engine import URL
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite.chunk_splitter import split_chunks
from raglite.database_models import Chunk, ChunkANNIndex, Document, create_database_engine
from raglite.markdown_converter import document_to_markdown
from raglite.proposition_extractor import extract_propositions
from raglite.sentence_splitter import split_sentences
from raglite.string_embedder import embed_strings


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
        chunks = split_chunks(sentences)
        pbar.update(1)
    # Process the chunks into propositions and store them in the database.
    with Session(engine) as session:
        # Add the document to the document table.
        document_record = Document.from_path(doc_path)
        if session.get(Document, document_record.id) is None:
            session.add(document_record)
            session.commit()
        # Add the document chunks to the chunk table.
        headers = ""
        for i, chunk in enumerate(
            tqdm(chunks, desc="Extracting propositions", unit="chunk", dynamic_ncols=True)
        ):
            # Initialise the chunk record.
            chunk_record = Chunk.from_body(
                document_id=document_record.id,
                index=i,
                body=chunk,
                headers=headers,
            )
            # Update the Markdown headers with those of this chunk.
            headers = chunk_record.extract_headers()
            # Continue if we've already inserted this chunk.
            if session.get(Chunk, chunk_record.id) is not None:
                continue
            # Extract propositions and compute their embeddings.
            chunk_record.propositions = extract_propositions(
                document_topic=headers.splitlines()[0].strip("# "),
                chunk_headers=chunk_record.headers,
                chunk_body=chunk_record.body,
            )
            chunk_record.proposition_embeddings = embed_strings(chunk_record.propositions)
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
            X_unindexed = np.vstack([chunk.proposition_embeddings for chunk in unindexed_chunks])  # noqa: N806
            if num_chunks_indexed == 0:
                chunk_ann_index.index = NNDescent(X_unindexed, metric="cosine", compressed=True)
            else:
                chunk_ann_index.index.prepare()
                chunk_ann_index.index.update(X_unindexed)
            chunk_ann_index.chunk_sizes.extend(
                [len(chunk.propositions) for chunk in unindexed_chunks]
            )
            pbar.update(num_chunks_unindexed)
            # Store the updated chunk ANN index.
            session.add(chunk_ann_index)
            session.commit()
