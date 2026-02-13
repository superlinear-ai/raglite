"""DuckDB or PostgreSQL database tables for RAGLite."""

import contextlib
import datetime
import json
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from markdown_it import MarkdownIt
from packaging import version
from pydantic import ConfigDict, PrivateAttr
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ProgrammingError
from sqlmodel import (
    JSON,
    Column,
    Field,
    Integer,
    Relationship,
    Sequence,
    Session,
    SQLModel,
    create_engine,
    text,
)

from raglite._config import RAGLiteConfig
from raglite._litellm import get_embedding_dim
from raglite._markdown import document_to_markdown
from raglite._typing import (
    ChunkId,
    DocumentId,
    Embedding,
    EvalId,
    FloatMatrix,
    FloatVector,
    IndexId,
    MetadataValue,
    PickledObject,
)

MetadataJSON = JSON().with_variant(JSONB(), "postgresql")


def _adapt_metadata(metadata: Any) -> dict[str, MetadataValue | list[MetadataValue]]:
    """Adapt metadata to the format expected by the database."""
    if not metadata:
        return {}
    return {k: v if isinstance(v, list) else [v] for k, v in metadata.items()}


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
    metadata_: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("metadata", MetadataJSON)
    )

    # Document content is not stored in the database, but is accessible via `document.content`.
    _content: str | None = PrivateAttr()

    # Add relationships so we can access document.chunks and document.evals.
    chunks: list["Chunk"] = Relationship(back_populates="document", cascade_delete=True)
    evals: list["Eval"] = Relationship(back_populates="document", cascade_delete=True)

    def __init__(self, **kwargs: Any) -> None:
        # Workaround for https://github.com/fastapi/sqlmodel/issues/149.
        super().__init__(**kwargs)
        self.content = kwargs.get("content")

    @property
    def content(self) -> str | None:
        return self._content

    @content.setter
    def content(self, value: str | None) -> None:
        self._content = value

    @property
    def front_matter(self) -> str:
        """Return this document's metadata formatted as YAML front matter."""
        metadata_lines = "\n".join(
            f"{key}: {value}" for key, value in self.metadata_.items() if value is not None
        )
        return f"---\n{metadata_lines}\n---" if metadata_lines else ""

    @staticmethod
    def from_path(
        doc_path: Path,
        *,
        id: DocumentId | None = None,  # noqa: A002
        url: str | None = None,
        **kwargs: Any,
    ) -> "Document":
        """Create a document from a file path.

        Parameters
        ----------
        doc_path
            The document's file path.
        id
            The document id to use. If not provided, a hash of the document's content is used.
        url
            The URL of the document, if available.
        kwargs
            Any additional metadata to store.

        Returns
        -------
        Document
            A document.
        """
        metadata = {
            "filename": doc_path.name,
            "uri": id,
            "url": url,
            "size": doc_path.stat().st_size,
            "created": doc_path.stat().st_ctime,
            "modified": doc_path.stat().st_mtime,
            **kwargs,
        }
        # Ensure all metadata values are lists
        metadata = _adapt_metadata(metadata)
        # Create the document instance.
        return Document(
            id=id if id is not None else hash_bytes(doc_path.read_bytes()),
            filename=doc_path.name,
            url=url,
            metadata_=metadata,
            content=document_to_markdown(doc_path),
        )

    @staticmethod
    def from_text(
        content: str,
        *,
        id: DocumentId | None = None,  # noqa: A002
        url: str | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> "Document":
        """Create a document from text content.

        Parameters
        ----------
        content
            The document's content as a text/plain or text/markdown string.
        id
            The document id to use. If not provided, a hash of the document's content is used.
        filename
            The document filename to use. If not provided, the first line of the content is used.
        url
            The URL of the document, if available.
        kwargs
            Any additional metadata to store.

        Returns
        -------
        Document
            A document.
        """
        # Extract the first line of the content.
        first_line = content.strip().split("\n", 1)[0].strip()
        if len(first_line) > 80:  # noqa: PLR2004
            first_line = f"{first_line[:80]}..."

        metadata = {
            "filename": filename or first_line,
            "uri": id,
            "url": url,
            "size": len(content.encode()),
            **kwargs,
        }
        # Ensure all metadata values are lists
        metadata = _adapt_metadata(metadata)
        # Create the document instance.
        return Document(
            id=id if id is not None else hash_bytes(content.encode()),
            filename=filename or first_line,
            url=url,
            metadata_=metadata,
            content=content,
        )

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, filename={self.filename!r})"


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
    metadata_: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("metadata", MetadataJSON)
    )

    # Add relationships so we can access chunk.document and chunk.embeddings.
    document: Document = Relationship(back_populates="chunks")
    embeddings: list["ChunkEmbedding"] = Relationship(back_populates="chunk", cascade_delete=True)

    @staticmethod
    def from_body(
        document: Document, index: int, body: str, headings: str = "", **kwargs: Any
    ) -> "Chunk":
        """Create a chunk from Markdown."""
        return Chunk(
            id=hash_bytes(f"{document.id}-{index}".encode()),
            document_id=document.id,
            index=index,
            headings=Chunk.truncate_headings(headings, body),
            body=body,
            metadata_=_adapt_metadata(
                {"filename": document.filename, "url": document.url, **kwargs}
            ),
        )

    @staticmethod
    def extract_heading_lines(doc: str, leading_only: bool = False) -> list[str]:  # noqa: FBT001, FBT002
        """Extract the leading or final state of the Markdown headings of a document."""
        md = MarkdownIt()
        heading_lines = [""] * 6
        level = None
        for token in md.parse(doc):
            if token.type == "heading_open":
                level = int(token.tag[1]) if 1 <= int(token.tag[1]) <= 6 else None  # noqa: PLR2004
            elif token.type == "heading_close":
                level = None
            elif level is not None:
                heading_content = token.content.strip().replace("\n", " ")
                heading_lines[level - 1] = ("#" * level) + " " + heading_content
                heading_lines[level:] = [""] * len(heading_lines[level:])
            elif leading_only and level is None and token.content and not token.content.isspace():
                break
        return heading_lines

    @staticmethod
    def truncate_headings(headings: str, body: str) -> str:
        """Truncate the contextual headings given the chunk's leading headings (if present)."""
        heading_lines = Chunk.extract_heading_lines(headings)
        leading_body_heading_lines = Chunk.extract_heading_lines(body, leading_only=True)
        level = next((i + 1 for i, line in enumerate(leading_body_heading_lines) if line), None)
        if level:
            heading_lines[level - 1 :] = [""] * len(heading_lines[level - 1 :])
        headings = "\n".join([heading for heading in heading_lines if heading])
        return headings

    def extract_headings(self) -> str:
        """Extract Markdown headings from the chunk, starting from the contextual headings."""
        heading_lines = self.extract_heading_lines(self.headings + "\n\n" + self.body)
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
    def front_matter(self) -> str:
        """Return this chunk's front matter."""
        # Compose the chunk metadata from the filename and URL.
        metadata = "\n".join(
            f"{key}: {self.metadata_.get(key)}"
            for key in ("filename", "url", "uri")
            if self.metadata_.get(key)
        )
        if not metadata:
            return ""
        # Return the chunk metadata in the YAML front matter format [1].
        # [1] https://jekyllrb.com/docs/front-matter/
        front_matter = f"---\n{metadata}\n---"
        return front_matter

    @property
    def content(self) -> str:
        """Return this chunk's front matter, contextual heading, and body."""
        return f"{self.front_matter}\n\n{self.headings.strip()}\n\n{self.body.strip()}".strip()

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
                f"<source>{self.document.url or self.document.filename}</source>",
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
            "source": self.document.url or self.document.filename,
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
        """Return this chunk span's front matter, contextual heading, and chunk bodies."""
        front_matter = self.chunks[0].front_matter if self.chunks else ""
        heading = self.chunks[0].headings.strip() if self.chunks else ""
        bodies = "".join(chunk.body for chunk in self.chunks)
        return f"{front_matter}\n\n{heading}\n\n{bodies}".strip()

    def __str__(self) -> str:
        """Return this chunk span's content."""
        return self.content


# We manually create an auto-incrementing sequence for `chunk_embedding.id` here because DuckDB
# doesn't support `id SERIAL` out of the box.
chunk_embedding_id_seq = Sequence("chunk_embedding_id_seq")


class ChunkEmbedding(SQLModel, table=True):
    """A (sub-)chunk embedding."""

    __tablename__ = "chunk_embedding"

    # Enable Embedding columns.
    model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]

    # Table columns.
    id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            chunk_embedding_id_seq,
            primary_key=True,
            server_default=chunk_embedding_id_seq.next_value(),
        ),
    )
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
        with Session(create_database_engine(config)) as session:
            index_metadata_record = session.get(IndexMetadata, id_)
            if index_metadata_record is None:
                return None
        return index_metadata_record.metadata_

    @staticmethod
    def get(id_: str = "default", *, config: RAGLiteConfig | None = None) -> dict[str, Any]:
        metadata = IndexMetadata._get(id_, config=config) or {}
        return metadata


class Metadata(SQLModel, table=True):
    """A table for metadata values, linked to field names."""

    __tablename__ = "metadata"

    name: str = Field(..., primary_key=True)
    values: list[MetadataValue] = Field(
        default_factory=list, sa_column=Column("metadata", MetadataJSON)
    )


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
    metadata_: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("metadata", MetadataJSON)
    )

    # Add relationship so we can access eval.document.
    document: Document = Relationship(back_populates="evals")

    @staticmethod
    def from_chunks(
        question: str, contexts: list[Chunk], ground_truth: str, **kwargs: Any
    ) -> "Eval":
        """Create a chunk from Markdown."""
        document_id = contexts[0].document_id
        chunk_ids = [context.id for context in contexts]
        # Ensure all metadata values from kwargs are lists
        processed_kwargs = _adapt_metadata(kwargs)
        return Eval(
            id=hash_bytes(f"{document_id}-{chunk_ids}-{question}".encode()),
            document_id=document_id,
            chunk_ids=chunk_ids,
            question=question,
            contexts=[str(context) for context in contexts],
            ground_truth=ground_truth,
            metadata_=processed_kwargs,
        )


@lru_cache(maxsize=1)
def create_database_engine(config: RAGLiteConfig | None = None) -> Engine:  # noqa: C901, PLR0912, PLR0915
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
    elif db_backend == "duckdb":
        with contextlib.suppress(Exception):
            if db_url.database and db_url.database != ":memory:":
                Path(db_url.database).parent.mkdir(parents=True, exist_ok=True)
    else:
        error_message = "RAGLite only supports DuckDB or PostgreSQL."
        raise ValueError(error_message)
    # Create the engine.
    engine = create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)
    # Install database extensions.
    if db_backend == "postgresql":
        with Session(engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
    elif db_backend == "duckdb":
        with Session(engine) as session:
            session.execute(text("INSTALL fts; LOAD fts;"))
            session.execute(text("INSTALL vss; LOAD vss;"))
            session.commit()
    # Get the embedding dimension.
    embedding_dim = get_embedding_dim(config)
    # Create all SQLModel tables.
    ChunkEmbedding.set_embedding_dim(embedding_dim)
    SQLModel.metadata.create_all(engine)
    # Create backend-specific indexes.
    oversample = round(4 * config.chunk_max_size / RAGLiteConfig.chunk_max_size)
    ef_search = (10 * 4) * oversample  # This is (# reranked results) * oversampling factor.
    if db_backend == "postgresql":
        with Session(engine) as session:
            # Create a keyword search index with `tsvector`.
            session.execute(
                text("""
                CREATE INDEX IF NOT EXISTS keyword_search_chunk_index ON chunk USING GIN (to_tsvector('simple', body));
                """)
            )
            # Create a vector search index with `pgvector`.
            metrics = {"cosine": "cosine", "dot": "ip", "l1": "l1", "l2": "l2"}
            create_vector_index_sql = f"""
                CREATE INDEX IF NOT EXISTS vector_search_chunk_index ON chunk_embedding
                USING hnsw (
                    (embedding::halfvec({embedding_dim}))
                    halfvec_{metrics[config.vector_search_distance_metric]}_ops
                );
                SET hnsw.ef_search = {ef_search};
            """
            # Enable iterative scan for pgvector v0.8.0 and up.
            pgvector_version = session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one()
            if pgvector_version and version.parse(pgvector_version) >= version.parse("0.8.0"):
                create_vector_index_sql += f"\nSET hnsw.iterative_scan = {'relaxed_order' if config.reranker else 'strict_order'};"
            session.execute(text(create_vector_index_sql))
            # Create a metadata search index if the metadata column is JSONB.
            metadata_column_type = session.execute(
                text("""
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = 'chunk' AND column_name = 'metadata'
                """)
            ).scalar_one_or_none()
            if metadata_column_type and metadata_column_type.lower() == "jsonb":
                session.execute(
                    text("""
                    CREATE INDEX IF NOT EXISTS metadata_gin_index ON chunk USING GIN (metadata jsonb_path_ops);
                    """)
                )
            session.commit()
    elif db_backend == "duckdb":
        with Session(engine) as session:
            # Create a keyword search index with FTS if it is missing or out of date. DuckDB does
            # not automatically update the FTS index when the table is modified, so we take the
            # opportunity to update it here on engine creation.
            num_chunks = session.execute(text("SELECT COUNT(*) FROM chunk")).scalar_one()
            try:
                num_indexed_chunks = session.execute(
                    text("SELECT COUNT(*) FROM fts_main_chunk.docs")
                ).scalar_one()
            except ProgrammingError:
                num_indexed_chunks = 0
            if num_indexed_chunks == 0 or num_indexed_chunks != num_chunks:
                session.execute(
                    text("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1);")
                )
            # Create a vector search index with VSS if it doesn't exist.
            session.execute(
                text(f"""
                SET hnsw_ef_search = {ef_search};
                SET hnsw_enable_experimental_persistence = true;
                """)
            )
            vss_index_exists = session.execute(
                text("""
                SELECT COUNT(*) > 0
                FROM duckdb_indexes()
                WHERE schema_name = current_schema()
                AND table_name = 'chunk_embedding'
                AND index_name = 'vector_search_chunk_index'
                """)
            ).scalar_one()
            if not vss_index_exists:
                metrics = {"cosine": "cosine", "dot": "ip", "l2": "l2sq"}
                create_vector_index_sql = f"""
                    CREATE INDEX vector_search_chunk_index
                    ON chunk_embedding
                    USING HNSW (embedding)
                    WITH (metric = '{metrics[config.vector_search_distance_metric]}');
                """
                session.execute(text(create_vector_index_sql))
            session.commit()
    return engine
