"""Fixtures for the tests."""

import pytest
from llama_cpp import Llama, LlamaRAMCache  # type: ignore[attr-defined]

from raglite import RAGLiteConfig


@pytest.fixture()
def simple_config() -> RAGLiteConfig:
    """Create a lightweight in-memory config for testing."""
    # Use a lightweight LLM.
    llm = Llama.from_pretrained(
        repo_id="bartowski/Phi-3.1-mini-4k-instruct-GGUF",  # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        filename="*Q4_K_M.gguf",
        n_ctx=4096,  # 0 = Use the model's context size (default is 512).
        n_gpu_layers=-1,  # -1 = Offload all layers to the GPU (default is 0).
        verbose=False,
    )
    llm.set_cache(LlamaRAMCache())
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
    config = RAGLiteConfig(llm=llm, embedder=embedder, db_url=db_url)
    return config
