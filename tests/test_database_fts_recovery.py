"""Regression tests for DuckDB initialization with a missing FTS index."""

from pathlib import Path

import pytest
from sqlmodel import Session, text

from raglite import RAGLiteConfig, keyword_search
from raglite._database import Chunk, Document, create_database_engine


@pytest.mark.parametrize("in_memory", [False, True])
def test_create_duckdb_without_fts_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, in_memory: bool
) -> None:
    """Initialize and rebuild missing search indexes without an aborted transaction."""
    monkeypatch.setattr("raglite._database.get_embedding_dim", lambda _: 3)
    config = RAGLiteConfig(
        db_url="duckdb:///:memory:" if in_memory else f"duckdb:///{tmp_path.as_posix()}/test.db",
        llm="unused",
        embedder="unused",
        reranker=None,
    )
    create_database_engine.cache_clear()
    engine = create_database_engine(config)
    try:
        with Session(engine) as session:
            assert (
                session.execute(text("SELECT COUNT(*) FROM fts_main_chunk.docs")).scalar_one() == 0
            )
            assert (
                session.execute(
                    text(
                        "SELECT COUNT(*) FROM duckdb_indexes() WHERE index_name = 'vector_search_chunk_index'"
                    )
                ).scalar_one()
                == 1
            )
            document = Document.from_text("Nebula star formation", id="astronomy")
            session.add(document)
            session.commit()
            chunk = Chunk.from_body(document, 0, "Nebula star formation")
            session.add(chunk)
            session.commit()
            chunk_id = chunk.id
            session.execute(text("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1)"))
            session.commit()
        assert keyword_search("nebula", config=config)[0] == [chunk_id]
        if not in_memory:
            with Session(engine) as session:
                session.execute(text("PRAGMA drop_fts_index('chunk')"))
                session.commit()
            engine.dispose()
            create_database_engine.cache_clear()
            engine = create_database_engine(config)
            assert keyword_search("nebula", config=config)[0] == [chunk_id]
    finally:
        create_database_engine.cache_clear()
        engine.dispose()
