"""PostgreSQL or SQLite database tables for RAGLite."""

import contextlib
import datetime
import json
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape

import numpy as np
from markdown_it import MarkdownIt
from packaging import version
from packaging.version import Version
from pydantic import ConfigDict
from sqlalchemy.engine import Engine, make_url
from sqlmodel import JSON, Column, Field, Relationship, Session, SQLModel, create_engine, text

from raglite._config import RAGLiteConfig
from raglite._litellm import get_embedding_dim
from raglite._typing import (
    ChunkId,
    DocumentId,
    Embedding,
    EvalId,
    FloatMatrix,
    FloatVector,
    IndexId,
    PickledObject,
)


def hash_bytes(data: bytes, max_len: int = 16) -> str:
    """Hash bytes to a hexadecimal string."""
    return sha256(data, usedforsecurity=False).hexdigest()[:max_len]


class Document(SQLModel, table=True):
    """A document."""

    # Enable JSON columns.
    model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]

    # Table columns.
    id: DocumentId = Field(..., primary_key=True)
    filename: str
    url: str | None = Field(default=None)
    metadata_: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))

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


class Chunk(SQLModel, table=True):
    """A document chunk."""

    # Enable JSON columns.
    model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]

    # Table columns.
    id: ChunkId = Field(..., primary_key=True)
    document_id: DocumentId = Field(..., foreign_key="document.id", index=True)
    index: int = Field(..., index=True)
    headings: str
    body: str
    metadata_: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))

    # Add relationships so we can access chunk.document and chunk.embeddings.
    document: Document = Relationship(back_populates="chunks")
    embeddings: list["ChunkEmbedding"] = Relationship(back_populates="chunk")

    @staticmethod
    def from_body(
        document_id: DocumentId, index: int, body: str, headings: str = "", **kwargs: Any
    ) -> "Chunk":
        """Create a chunk from Markdown."""
        return Chunk(
            id=hash_bytes(body.encode()),
            document_id=document_id,
            index=index,
            headings=headings,
            body=body,
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

    @property
    def embedding_matrix(self) -> FloatMatrix:
        """Return this chunk's multi-vector embedding matrix."""
        # Uses the relationship chunk.embeddings to access the chunk_embedding table.
        return np.vstack([embedding.embedding[np.newaxis, :] for embedding in self.embeddings])

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "document_id": self.document_id,
                "index": self.index,
                "headings": self.headings,
                "body": self.body[:100],
                "metadata": self.metadata_,
            },
            indent=4,
        )

    @property
    def content(self) -> str:
        """Return this chunk's contextual heading and body."""
        return f"{self.headings.strip()}\n\n{self.body.strip()}".strip()

    def __str__(self) -> str:
        """Return this chunk's content."""
        return self.content


@dataclass
class ChunkSpan:
    """A consecutive sequence of chunks from a single document."""

    chunks: list[Chunk]
    document: Document = field(init=False)

    def __post_init__(self) -> None:
        """Set the document field."""
        if self.chunks:
            self.document = self.chunks[0].document

    def to_xml(self, index: int | None = None) -> str:
        """Convert this chunk span to an XML representation.

        The XML representation follows Anthropic's best practices [1].

        [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
        """
        if not self.chunks:
            return ""
        index_attribute = f' index="{index}"' if index is not None else ""
        xml_document = "\n".join(
            [
                f'<document{index_attribute} id="{self.document.id}">',
                f"<source>{self.document.url if self.document.url else self.document.filename}</source>",
                f'<span from_chunk_id="{self.chunks[0].id}" to_chunk_id="{self.chunks[-1].id}">',
                f"<headings>\n{escape(self.chunks[0].headings.strip())}\n</headings>",
                f"<content>\n{escape(''.join(chunk.body for chunk in self.chunks).strip())}\n</content>",
                "</span>",
                "</document>",
            ]
        )
        return xml_document

    def to_json(self, index: int | None = None) -> str:
        """Convert this chunk span to a JSON representation.

        The JSON representation follows Anthropic's best practices [1].

        [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
        """
        if not self.chunks:
            return "{}"
        index_attribute = {"index": index} if index is not None else {}
        json_document = {
            **index_attribute,
            "id": self.document.id,
            "source": self.document.url if self.document.url else self.document.filename,
            "span": {
                "from_chunk_id": self.chunks[0].id,
                "to_chunk_id": self.chunks[-1].id,
                "headings": self.chunks[0].headings.strip(),
                "content": "".join(chunk.body for chunk in self.chunks).strip(),
            },
        }
        return json.dumps(json_document)

    @property
    def content(self) -> str:
        """Return this chunk span's contextual heading and chunk bodies."""
        heading = self.chunks[0].headings.strip() if self.chunks else ""
        bodies = "".join(chunk.body for chunk in self.chunks)
        return f"{heading}\n\n{bodies}".strip()

    def __str__(self) -> str:
        """Return this chunk span's content."""
        return self.content


class ChunkEmbedding(SQLModel, table=True):
    """A (sub-)chunk embedding."""

    __tablename__ = "chunk_embedding"

    # Enable Embedding columns.
    model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]

    # Table columns.
    id: int = Field(..., primary_key=True)
    chunk_id: ChunkId = Field(..., foreign_key="chunk.id", index=True)
    embedding: FloatVector = Field(..., sa_column=Column(Embedding(dim=-1)))

    # Add relationship so we can access embedding.chunk.
    chunk: Chunk = Relationship(back_populates="embeddings")

    @classmethod
    def set_embedding_dim(cls, dim: int) -> None:
        """Modify the embedding column's dimension after class definition."""
        cls.__table__.c["embedding"].type.dim = dim  # type: ignore[attr-defined]


class IndexMetadata(SQLModel, table=True):
    """Vector and keyword search index metadata."""

    __tablename__ = "index_metadata"

    # Enable PickledObject columns.
    model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]

    # Table columns.
    id: IndexId = Field(..., primary_key=True)
    version: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    metadata_: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("metadata", PickledObject)
    )

    @staticmethod
    @lru_cache(maxsize=4)
    def _get(id_: str, *, config: RAGLiteConfig | None = None) -> dict[str, Any] | None:
        engine = create_database_engine(config)
        with Session(engine) as session:
            index_metadata_record = session.get(IndexMetadata, id_)
            if index_metadata_record is None:
                return None
        return index_metadata_record.metadata_

    @staticmethod
    def get(id_: str = "default", *, config: RAGLiteConfig | None = None) -> dict[str, Any]:
        metadata = IndexMetadata._get(id_, config=config) or {}
        return metadata


class Eval(SQLModel, table=True):
    """A RAG evaluation example."""

    __tablename__ = "eval"

    # Enable JSON columns.
    model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]

    # Table columns.
    id: EvalId = Field(..., primary_key=True)
    document_id: DocumentId = Field(..., foreign_key="document.id", index=True)
    chunk_ids: list[ChunkId] = Field(default_factory=list, sa_column=Column(JSON))
    question: str
    contexts: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    ground_truth: str
    metadata_: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))

    # Add relationship so we can access eval.document.
    document: Document = Relationship(back_populates="evals")

    @staticmethod
    def from_chunks(
        question: str, contexts: list[Chunk], ground_truth: str, **kwargs: Any
    ) -> "Eval":
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


def _pgvector_version(session: Session) -> Version:
    try:
        result = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        pgvector_version = version.parse(cast(str, result.scalar_one()))
    except Exception as e:
        error_message = "Unable to parse pgvector version, is pgvector installed?"
        raise ValueError(error_message) from e
    return pgvector_version


@lru_cache(maxsize=1)
def create_database_engine(config: RAGLiteConfig | None = None) -> Engine:
    """Create a database engine and initialize it."""
    # Parse the database URL and validate that the database backend is supported.
    config = config or RAGLiteConfig()
    db_url = make_url(config.db_url)
    db_backend = db_url.get_backend_name()
    # Update database configuration.
    connect_args = {}
    if db_backend == "postgresql":
        # Select the pg8000 driver if not set (psycopg2 is the default), and prefer SSL.
        if "+" not in db_url.drivername:
            db_url = db_url.set(drivername="postgresql+pg8000")
        # Support setting the sslmode for pg8000.
        if "pg8000" in db_url.drivername and "sslmode" in db_url.query:
            query = dict(db_url.query)
            if query.pop("sslmode") != "disable":
                connect_args["ssl_context"] = True
            db_url = db_url.set(query=query)
    elif db_backend == "sqlite":
        # Optimize SQLite performance.
        pragmas = {"journal_mode": "WAL", "synchronous": "NORMAL"}
        db_url = db_url.update_query_dict(pragmas, append=True)
        # Create the database path if it doesn't exist.
        with contextlib.suppress(Exception):
            Path(db_url.database).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
    else:
        error_message = "RAGLite only supports PostgreSQL and SQLite."
        raise ValueError(error_message)
    # Create the engine.
    engine = create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)
    # Install database extensions.
    if db_backend == "postgresql":
        with Session(engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
    # Get the embedding dimension.
    embedding_dim = get_embedding_dim(config)
    # Create all SQLModel tables.
    ChunkEmbedding.set_embedding_dim(embedding_dim)
    SQLModel.metadata.create_all(engine)
    # Create backend-specific indexes.
    if db_backend == "postgresql":
        # Create a keyword search index with `tsvector` and a vector search index with `pgvector`.
        with Session(engine) as session:
            metrics = {"cosine": "cosine", "dot": "ip", "euclidean": "l2", "l1": "l1", "l2": "l2"}
            session.execute(
                text("""
                CREATE INDEX IF NOT EXISTS keyword_search_chunk_index ON chunk USING GIN (to_tsvector('simple', body));
                """)
            )
            create_vector_index_sql = f"""
                CREATE INDEX IF NOT EXISTS vector_search_chunk_index ON chunk_embedding
                USING hnsw (
                    (embedding::halfvec({embedding_dim}))
                    halfvec_{metrics[config.vector_search_index_metric]}_ops
                );
                SET hnsw.ef_search = {20 * 4 * 8};
            """
            # Enable iterative scan for pgvector v0.8.0 and up.
            pgvector_version = _pgvector_version(session)
            if pgvector_version and pgvector_version >= version.parse("0.8.0"):
                create_vector_index_sql += f"\nSET hnsw.iterative_scan = {'relaxed_order' if config.reranker else 'strict_order'};"
            session.execute(text(create_vector_index_sql))
            session.commit()
    elif db_backend == "sqlite":
        # Create a virtual table for keyword search on the chunk table.
        # We use the chunk table as an external content table [1] to avoid duplicating the data.
        # [1] https://www.sqlite.org/fts5.html#external_content_tables
        with Session(engine) as session:
            session.execute(
                text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS keyword_search_chunk_index USING fts5(body, content='chunk', content_rowid='rowid');
                """)
            )
            session.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS keyword_search_chunk_index_auto_insert AFTER INSERT ON chunk BEGIN
                    INSERT INTO keyword_search_chunk_index(rowid, body) VALUES (new.rowid, new.body);
                END;
                """)
            )
            session.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS keyword_search_chunk_index_auto_delete AFTER DELETE ON chunk BEGIN
                    INSERT INTO keyword_search_chunk_index(keyword_search_chunk_index, rowid, body) VALUES('delete', old.rowid, old.body);
                END;
                """)
            )
            session.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS keyword_search_chunk_index_auto_update AFTER UPDATE ON chunk BEGIN
                    INSERT INTO keyword_search_chunk_index(keyword_search_chunk_index, rowid, body) VALUES('delete', old.rowid, old.body);
                    INSERT INTO keyword_search_chunk_index(rowid, body) VALUES (new.rowid, new.body);
                END;
                """)
            )
            session.commit()
    return engine
