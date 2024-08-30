"""Fixtures for the tests."""

import socket

import pytest
from llama_cpp import Llama
from sqlalchemy import create_engine, text

from raglite import RAGLiteConfig


def is_postgres_running() -> bool:
    """Check if PostgreSQL is running."""
    try:
        with socket.create_connection(("postgres", 5432), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("sqlite:///:memory:", id="SQLite"),
        pytest.param(
            "postgresql+pg8000://raglite_user:raglite_password@postgres:5432/postgres",
            id="PostgreSQL",
            marks=pytest.mark.skipif(not is_postgres_running(), reason="PostgreSQL is not running"),
        ),
    ],
)
def simple_config(request: pytest.FixtureRequest) -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing SQLite and PostgreSQL."""
    # Use a lightweight embedder.
    embedder = Llama.from_pretrained(
        repo_id="ChristianAzinn/snowflake-arctic-embed-xs-gguf",  # https://github.com/Snowflake-Labs/arctic-embed
        filename="*f16.GGUF",
        n_ctx=0,  # 0 = Use the model's context size (default is 512).
        n_gpu_layers=-1,  # -1 = Offload all layers to the GPU (default is 0).
        verbose=False,
        embedding=True,
    )
    # Yield a SQLite config.
    if "sqlite" in request.param:
        sqlite_config = RAGLiteConfig(embedder=embedder, db_url=request.param)
        return sqlite_config
    # Yield a PostgreSQL config.
    if "postgresql" in request.param:
        engine = create_engine(request.param, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(text("DROP DATABASE IF EXISTS raglite_test"))
            conn.execute(text("CREATE DATABASE raglite_test"))
        postgresql_config = RAGLiteConfig(
            embedder=embedder, db_url=request.param.replace("/postgres", "/raglite_test")
        )
        return postgresql_config
    raise ValueError
