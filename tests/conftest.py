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
    """Reset the PostgreSQL database."""
    if is_postgres_running():
        engine = create_engine(POSTGRES_URL, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            for variant in ["local", "remote"]:
                conn.execute(text(f"DROP DATABASE IF EXISTS raglite_test_{variant}"))
                conn.execute(text(f"CREATE DATABASE raglite_test_{variant}"))


@pytest.fixture(scope="session")
def duckdb_url() -> Generator[str, None, None]:
    """Create a temporary DuckDB database file and return the database URL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / "raglite_test.db"
        yield f"duckdb:///{db_file}"


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("duckdb", id="duckdb"),
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
        request.getfixturevalue("duckdb_url") if request.param == "duckdb" else request.param
    )
    return db_url


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            (
                "llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192",
                "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@512",  # More context degrades performance.
            ),
            id="qwen3_4B-bge_m3",
        ),
        pytest.param(
            ("gpt-4o-mini", "text-embedding-3-small"),
            id="gpt_4o_mini-text_embedding_3_small",
            marks=pytest.mark.skipif(not is_openai_available(), reason="OpenAI API key is not set"),
        ),
    ],
)
def llm_embedder(request: pytest.FixtureRequest) -> str:
    """Get an LLM and embedder pair to test RAGLite with."""
    llm_embedder: str = request.param
    return llm_embedder


@pytest.fixture(scope="session")
def llm(llm_embedder: tuple[str, str]) -> str:
    """Get an LLM to test RAGLite with."""
    llm, _ = llm_embedder
    return llm


@pytest.fixture(scope="session")
def embedder(llm_embedder: tuple[str, str]) -> str:
    """Get an embedder to test RAGLite with."""
    _, embedder = llm_embedder
    return embedder


@pytest.fixture(scope="session")
def raglite_test_config(database: str, llm: str, embedder: str) -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing DuckDB and PostgreSQL."""
    # Select the database based on the embedder.
    variant = "local" if embedder.startswith("llama-cpp-python") else "remote"
    if "postgres" in database:
        database = database.replace("/postgres", f"/raglite_test_{variant}")
    elif "duckdb" in database:
        database = database.replace(".db", f"_{variant}.db")
    # Create a RAGLite config for the given database and embedder.
    db_config = RAGLiteConfig(db_url=database, llm=llm, embedder=embedder)
    # Insert a document and update the index.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=db_config)
    return db_config
