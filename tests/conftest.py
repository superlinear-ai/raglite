"""Fixtures for the tests."""

import os
import socket
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Dict, List, Optional, Any

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
            (
                "llama-cpp-python/bartowski/Llama-3.2-3B-Instruct-GGUF/*Q4_K_M.gguf@4096",
                "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@1024",  # More context degrades performance.
            ),
            id="llama_3.2_3B-bge_m3",
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
    """Create a lightweight in-memory config for testing SQLite and PostgreSQL."""
    # Select the database based on the embedder.
    variant = "local" if embedder.startswith("llama-cpp-python") else "remote"
    if "postgres" in database:
        database = database.replace("/postgres", f"/raglite_test_{variant}")
    elif "sqlite" in database:
        database = database.replace(".sqlite", f"_{variant}.sqlite")
    # Create a RAGLite config for the given database and embedder.
    db_config = RAGLiteConfig(db_url=database, llm=llm, embedder=embedder)
    # Insert a document and update the index.
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    insert_document(doc_path, config=db_config)
    return db_config


@pytest.fixture
def golden_test_config(
    database: str, llm: str, embedder: str, request: pytest.FixtureRequest
) -> tuple[RAGLiteConfig, list, str]:
    """Create a config for golden tests with customizable documents and model.

    This fixture creates a fresh config without inserting any documents by default,
    and allows specifying:
    - documents: List of document paths to insert
    - test_cases: List of query/document pairs for testing
    - model: Model to use for creating the golden dataset

    Usage:
        @pytest.mark.parametrize(
            "golden_test_config",
            [
                {
                    "documents": ["tests/specrel.pdf", "tests/paul_graham_essay.txt"],
                    "test_cases": [
                        {"query": "What is relativity?", "document": "specrel.pdf"},
                        {"query": "What did Paul Graham do?", "document": "paul_graham_essay.txt"}
                    ],
                    "model": "gpt-4o"
                }
            ],
            indirect=True
        )
        def test_something(golden_test_config):
            config, test_cases, model = golden_test_config
            # Use the config, test_cases, and model
            ...
    """
    # Create a fresh config (similar to raglite_test_config but without inserting specrel.pdf)
    variant = "local" if embedder.startswith("llama-cpp-python") else "remote"
    db_url = database
    if "postgres" in database:
        db_url = database.replace("/postgres", f"/raglite_test_{variant}")
    elif "sqlite" in database:
        db_url = database.replace(".sqlite", f"_{variant}.sqlite")

    # Create a RAGLite config
    config = RAGLiteConfig(db_url=db_url, llm=llm, embedder=embedder)

    # Get customization parameters (if any)
    params = getattr(request, "param", {})

    # Insert documents if specified
    documents = params.get("documents", [])
    for doc_path in documents:
        insert_document(Path(doc_path), config=config)

    # Get test cases and model
    test_cases = params.get("test_cases", [])
    model = params.get("model", "gpt-4o-mini")

    return config, test_cases, model
