"""Fixtures for the tests."""

import os
import socket
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from raglite import RAGLiteConfig, insert_document

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
    """Reset the PostgreSQL and SQLite databases."""
    if is_postgres_running():
        engine = create_engine(POSTGRES_URL, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            for variant in ["local", "remote"]:
                conn.execute(text(f"DROP DATABASE IF EXISTS raglite_test_{variant}"))
                conn.execute(text(f"CREATE DATABASE raglite_test_{variant}"))


@pytest.fixture(scope="session")
def sqlite_url() -> Generator[str, None, None]:
    """Create a temporary SQLite database file and return the database URL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / "raglite_test.sqlite"
        yield f"sqlite:///{db_file}"


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            POSTGRES_URL,
            id="postgres",
            marks=pytest.mark.skipif(not is_postgres_running(), reason="PostgreSQL is not running"),
        ),
    ],
)
def database(request: pytest.FixtureRequest) -> str:
    """Get a database URL to test RAGLite with."""
    db_url: str = (
        request.getfixturevalue("sqlite_url") if request.param == "sqlite" else request.param
    )
    return db_url


@pytest.fixture(
    scope="session",
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


@pytest.fixture(scope="session")
def raglite_test_config(database: str, embedder: str) -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing SQLite and PostgreSQL."""
    # Select the database based on the embedder.
    variant = "local" if embedder.startswith("llama-cpp-python") else "remote"
    if "postgres" in database:
        database = database.replace("/postgres", f"/raglite_test_{variant}")
    elif "sqlite" in database:
        database = database.replace(".sqlite", f"_{variant}.sqlite")
    # Create a RAGLite config for the given database and embedder.
    db_config = RAGLiteConfig(db_url=database, embedder=embedder)
    # Insert a document and update the index.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=db_config)
    return db_config
