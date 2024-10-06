"""Fixtures for the tests."""

import os
import socket

import pytest
from sqlalchemy import create_engine, text

from raglite import RAGLiteConfig

POSTGRES_URL = "postgresql+pg8000://raglite_user:raglite_password@postgres:5432/postgres"


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


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reset the PostgreSQL database."""
    if is_postgres_running():
        engine = create_engine(POSTGRES_URL, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            for variant in ["local", "remote"]:
                conn.execute(text(f"DROP DATABASE IF EXISTS raglite_test_{variant}"))
                conn.execute(text(f"CREATE DATABASE raglite_test_{variant}"))


@pytest.fixture(
    params=[
        pytest.param("sqlite:///:memory:", id="sqlite"),
        pytest.param(
            POSTGRES_URL,
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
    params=[
        pytest.param(
            "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf",
            id="bge_m3",
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


@pytest.fixture
def raglite_test_config(database: str, embedder: str) -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing SQLite and PostgreSQL."""
    # Select the PostgreSQL database based on the embedder.
    if "postgres" in database:
        variant = "local" if embedder.startswith("llama-cpp-python") else "remote"
        database = database.replace("/postgres", f"/raglite_test_{variant}")
    # Create a RAGLite config for the given database and embedder.
    db_config = RAGLiteConfig(db_url=database, embedder=embedder)
    return db_config
