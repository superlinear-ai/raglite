"""Fixtures for the tests."""

import pytest
from llama_cpp import Llama

from raglite import RAGLiteConfig


@pytest.fixture
def simple_config() -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing."""
    # Use a lightweight embedder.
    embedder = Llama.from_pretrained(
        repo_id="ChristianAzinn/snowflake-arctic-embed-xs-gguf",  # https://github.com/Snowflake-Labs/arctic-embed
        filename="*f16.GGUF",
        n_ctx=0,  # 0 = Use the model's context size (default is 512).
        n_gpu_layers=-1,  # -1 = Offload all layers to the GPU (default is 0).
        verbose=False,
        embedding=True,
    )
    # Use an in-memory SQLite database.
    db_url = "sqlite:///:memory:"
    # Create the config.
    config = RAGLiteConfig(embedder=embedder, db_url=db_url)
    return config
