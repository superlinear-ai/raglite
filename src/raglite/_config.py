"""RAGLite config."""

import os
from dataclasses import dataclass

from llama_cpp import llama_supports_gpu_offload
from sqlalchemy.engine import URL

DEFAULT_LLM = (
    "llama-cpp-python/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/*Q4_K_M.gguf@8192"
    if llama_supports_gpu_offload()
    else "llama-cpp-python/bartowski/Phi-3.5-mini-instruct-GGUF/*Q4_K_M.gguf@4096"
)

DEFAULT_EMBEDDER = (
    "llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf"
    if llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 4  # noqa: PLR2004
    else "llama-cpp-python/yishan-wang/snowflake-arctic-embed-m-v1.5-Q8_0-GGUF/*q8_0.gguf"
)


@dataclass(frozen=True)
class RAGLiteConfig:
    """Configuration for RAGLite."""

    # Database config.
    db_url: str | URL = "sqlite:///raglite.sqlite"
    # LLM config used for generation.
    llm: str = DEFAULT_LLM
    llm_max_tries: int = 4
    # Embedder config used for indexing.
    embedder: str = DEFAULT_EMBEDDER
    embedder_normalize: bool = True
    embedder_sentence_window_size: int = 3
    # Chunk config used to partition documents into chunks.
    chunk_max_size: int = 1440  # Max number of characters per chunk.
    # Vector search config.
    vector_search_index_metric: str = "cosine"  # The query adapter supports "dot" and "cosine".
    vector_search_query_adapter: bool = True

    def __post_init__(self) -> None:
        # Late chunking with llama-cpp-python does not apply sentence windowing.
        if self.embedder.startswith("llama-cpp-python"):
            object.__setattr__(self, "embedder_sentence_window_size", 1)
