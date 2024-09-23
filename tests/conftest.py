"""Fixtures for the tests."""

import os
import socket

import pytest
from sqlalchemy import create_engine, text

from raglite import RAGLiteConfig


def is_postgres_running() -> bool:
    """Check if PostgreSQL is running."""
    try:
        with socket.create_connection(("postgres", 5432), timeout=1):
            return True
    except OSError:
        return False


def is_openai_available() -> bool:
    """Check if an OpenAI API key is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("sqlite:///:memory:", id="sqlite"),
        pytest.param(
            "postgresql+pg8000://raglite_user:raglite_password@postgres:5432/postgres",
            id="postgres",
            marks=pytest.mark.skipif(not is_postgres_running(), reason="PostgreSQL is not running"),
        ),
    ],
)
def database(request: pytest.FixtureRequest) -> str:
    """Get a database URL to test RAGLite with."""
    db_url: str = request.param
    return db_url


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            "llama-cpp-python/ChristianAzinn/snowflake-arctic-embed-xs-gguf/*f16.GGUF",
            id="snowflake_arctic_embed_xs",
        ),
        pytest.param(
            "text-embedding-3-small",
            id="openai_text_embedding_3_small",
            marks=pytest.mark.skipif(not is_openai_available(), reason="OpenAI API key is not set"),
        ),
    ],
)
def embedder(request: pytest.FixtureRequest) -> str:
    """Get an embedder model URL to test RAGLite with."""
    embedder: str = request.param
    return embedder


@pytest.fixture(scope="module")
def raglite_test_config(database: str, embedder: str) -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing SQLite and PostgreSQL."""
    # Yield a SQLite config.
    if "sqlite" in database:
        sqlite_config = RAGLiteConfig(embedder=embedder, db_url=database)
        return sqlite_config
    # Yield a PostgreSQL config.
    if "postgres" in database:
        # Reset the test database.
        engine = create_engine(database, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(text("DROP DATABASE IF EXISTS raglite_test"))
            conn.execute(text("CREATE DATABASE raglite_test"))
        # Create a PostgreSQL config.
        postgres_config = RAGLiteConfig(
            embedder=embedder,
            db_url=database.replace("/postgres", "/raglite_test"),
        )
        return postgres_config
    raise ValueError
