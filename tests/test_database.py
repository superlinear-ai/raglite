"""Test RAGLite's database engine creation."""

from pathlib import Path

from raglite import RAGLiteConfig
from raglite._database import create_database_engine


def test_in_memory_duckdb_creation(tmp_path: Path) -> None:
    """Test creating an in-memory DuckDB database."""
    config = RAGLiteConfig(db_url="duckdb:///:memory:")
    create_database_engine(config)


def test_repeated_duckdb_creation(tmp_path: Path) -> None:
    """Test creating the same DuckDB database engine twice."""
    duckdb_filepath = tmp_path / "test.db"
    config = RAGLiteConfig(db_url=f"duckdb:///{duckdb_filepath.as_posix()}")
    create_database_engine(config)
    create_database_engine.cache_clear()
    create_database_engine(config)
