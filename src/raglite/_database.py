"""SQLite tables for RAGLite."""

import io
import pickle
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from markdown_it import MarkdownIt
from pynndescent import NNDescent
from sqlalchemy.engine import URL, Dialect, Engine, make_url
from sqlalchemy.types import LargeBinary, TypeDecorator
from sqlmodel import JSON, Column, Field, Relationship, Session, SQLModel, create_engine, text


def hash_bytes(data: bytes, max_len: int = 16) -> str:
    """Hash bytes to a hexadecimal string."""
    return sha256(data, usedforsecurity=False).hexdigest()[:max_len]


class NumpyArray(TypeDecorator):
    """A NumPy array column type for SQLAlchemy."""

    impl = LargeBinary

    def process_bind_param(self, value: np.ndarray | None, dialect: Dialect) -> bytes | None:
        """Convert a NumPy array to bytes."""
        if value is None:
            return None
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False, fix_imports=False)
        return buffer.getvalue()

    def process_result_value(self, value: bytes | None, dialect: Dialect) -> np.ndarray | None:
        """Convert bytes to a NumPy array."""
        if value is None:
            return None
        return np.load(io.BytesIO(value), allow_pickle=False, fix_imports=False)


class PickledObject(TypeDecorator):
    """A pickled object column type for SQLAlchemy."""

    impl = LargeBinary

    def process_bind_param(self, value: object | None, dialect: Dialect) -> bytes | None:
        """Convert a Python object to bytes."""
        if value is None:
            return None
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)

    def process_result_value(self, value: bytes | None, dialect: Dialect) -> object | None:
        """Convert bytes to a Python object."""
        if value is None:
            return None
        return pickle.loads(value, fix_imports=False)  # noqa: S301


class Document(SQLModel, table=True):
    """A document."""

    id: str = Field(..., primary_key=True)
    filename: str
    url: str | None = Field(default=None)
    metadata_: dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))

    # Add relationships so we can access document.chunks and document.evals.
    chunks: list["Chunk"] = Relationship(back_populates="document")
    evals: list["Eval"] = Relationship(back_populates="document")

    @staticmethod
    def from_path(doc_path: Path, **kwargs: Any) -> "Document":
        """Create a document from a file path."""
        return Document(
            id=hash_bytes(doc_path.read_bytes()),
            filename=doc_path.name,
            metadata_={
                "size": doc_path.stat().st_size,
                "created": doc_path.stat().st_ctime,
                "modified": doc_path.stat().st_mtime,
                **kwargs,
            },
        )

    # Enable support for JSON columns.
    class Config:
        """Table configuration."""

        arbitrary_types_allowed = True


class Chunk(SQLModel, table=True):
    """A document chunk."""

    id: str = Field(..., primary_key=True)
    document_id: str = Field(..., foreign_key="document.id", index=True)
    index: int = Field(..., index=True)
    headings: str
    body: str
    multi_vector_embedding: np.ndarray = Field(..., sa_column=Column(NumpyArray))
    metadata_: dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))

    # Add relationship so we can access chunk.document.
    document: Document = Relationship(back_populates="chunks")

    @staticmethod
    def from_body(
        document_id: str,
        index: int,
        body: str,
        headings: str = "",
        multi_vector_embedding: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "Chunk":
        """Create a chunk from Markdown."""
        return Chunk(
            id=hash_bytes(body.encode()),
            document_id=document_id,
            index=index,
            headings=headings,
            body=body,
            multi_vector_embedding=multi_vector_embedding
            if multi_vector_embedding is not None
            else np.empty(0),
            metadata_=kwargs,
        )

    def extract_headings(self) -> str:
        """Extract Markdown headings from the chunk, starting from the current Markdown headings."""
        md = MarkdownIt()
        heading_lines = [""] * 10
        level = None
        for doc in (self.headings, self.body):
            for token in md.parse(doc):
                if token.type == "heading_open":
                    level = int(token.tag[1])
                elif token.type == "heading_close":
                    level = None
                elif level is not None:
                    heading_content = token.content.strip().replace("\n", " ")
                    heading_lines[level] = ("#" * level) + " " + heading_content
                    heading_lines[level + 1 :] = [""] * len(heading_lines[level + 1 :])
        headings = "\n".join([heading for heading in heading_lines if heading])
        return headings

    def __str__(self) -> str:
        """Context representation of this chunk."""
        return f"{self.headings.strip()}\n\n{self.body.strip()}".strip()

    def __hash__(self) -> int:
        return hash(self.id)

    # Enable support for JSON and NumpyArray columns.
    class Config:
        """Table configuration."""

        arbitrary_types_allowed = True


class ChunkANNIndex(SQLModel, table=True):
    """A chunk ANN index."""

    __tablename__ = "chunk_ann_index"

    id: str = Field(..., primary_key=True)
    chunk_sizes: list[int] = Field(default=[], sa_column=Column(JSON))
    index: NNDescent | None = Field(default=None, sa_column=Column(PickledObject))
    query_adapter: np.ndarray | None = Field(default=None, sa_column=Column(NumpyArray))
    metadata_: dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))

    # Enable support for JSON, PickledObject, and NumpyArray columns.
    class Config:
        """Table configuration."""

        arbitrary_types_allowed = True


class Eval(SQLModel, table=True):
    """A RAG evaluation example."""

    __tablename__ = "eval"

    id: str = Field(..., primary_key=True)
    document_id: str = Field(..., foreign_key="document.id", index=True)
    chunk_ids: list[str] = Field(default=[], sa_column=Column(JSON))
    question: str
    contexts: list[str] = Field(default=[], sa_column=Column(JSON))
    ground_truth: str
    metadata_: dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))

    # Add relationship so we can access eval.document.
    document: Document = Relationship(back_populates="evals")

    @staticmethod
    def from_chunks(
        question: str,
        contexts: list[Chunk],
        ground_truth: str,
        **kwargs: Any,
    ) -> "Chunk":
        """Create a chunk from Markdown."""
        document_id = contexts[0].document_id
        chunk_ids = [context.id for context in contexts]
        return Eval(
            id=hash_bytes(f"{document_id}-{chunk_ids}-{question}".encode()),
            document_id=document_id,
            chunk_ids=chunk_ids,
            question=question,
            contexts=[str(context) for context in contexts],
            ground_truth=ground_truth,
            metadata_=kwargs,
        )

    # Enable support for JSON columns.
    class Config:
        """Table configuration."""

        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def create_database_engine(db_url: str | URL = "sqlite:///raglite.sqlite") -> Engine:
    """Create a database engine and initialize it."""
    # Parse the database URL.
    db_url = make_url(db_url)
    assert db_url.get_backend_name() == "sqlite", "RAGLite currently only supports SQLite."
    # Optimize SQLite performance.
    pragmas = {"journal_mode": "WAL", "synchronous": "NORMAL"}
    db_url = db_url.update_query_dict(pragmas, append=True)
    # Create the engine.
    engine = create_engine(db_url)
    # Create all SQLModel tables.
    SQLModel.metadata.create_all(engine)
    # Create a virtual table for full-text search on the chunk table.
    # We use the chunk table as an external content table [1] to avoid duplicating the data.
    # [1] https://www.sqlite.org/fts5.html#external_content_tables
    with Session(engine) as session:
        session.exec(
            text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(body, content='chunk', content_rowid='rowid');
        """)
        )
        session.exec(
            text("""
        CREATE TRIGGER IF NOT EXISTS chunk_fts_auto_insert AFTER INSERT ON chunk BEGIN
            INSERT INTO chunk_fts(rowid, body) VALUES (new.rowid, new.body);
        END;
        """)
        )
        session.exec(
            text("""
        CREATE TRIGGER IF NOT EXISTS chunk_fts_auto_delete AFTER DELETE ON chunk BEGIN
            INSERT INTO chunk_fts(chunk_fts, rowid, body) VALUES('delete', old.rowid, old.body);
        END;
        """)
        )
        session.exec(
            text("""
        CREATE TRIGGER IF NOT EXISTS chunk_fts_auto_update AFTER UPDATE ON chunk BEGIN
            INSERT INTO chunk_fts(chunk_fts, rowid, body) VALUES('delete', old.rowid, old.body);
            INSERT INTO chunk_fts(rowid, body) VALUES (new.rowid, new.body);
        END;
        """)
        )
        session.commit()
    return engine
