"""RAGLite config."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaRAMCache  # type: ignore[attr-defined]
from sqlalchemy.engine import URL


@lru_cache(maxsize=1)
def default_llm() -> Llama:
    """Get default LLM."""
    llm = Llama.from_pretrained(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",  # https://github.com/meta-llama/llama-models
        filename="*Q4_K_M.gguf",
        n_ctx=8192,  # 0 = Use the model's context size (default is 512).
        n_gpu_layers=-1,  # -1 = Offload all layers to the GPU (default is 0).
        verbose=False,
    )
    llm.set_cache(LlamaRAMCache())
    return llm


@lru_cache(maxsize=1)
def default_embedder() -> Llama:
    """Get default embedder."""
    embedder = Llama.from_pretrained(
        repo_id="ChristianAzinn/snowflake-arctic-embed-l-gguf",  # https://github.com/Snowflake-Labs/arctic-embed
        filename="*f16.GGUF",
        n_ctx=0,  # 0 = Use the model's context size (default is 512).
        n_gpu_layers=-1,  # -1 = Offload all layers to the GPU (default is 0).
        verbose=False,
        embedding=True,
    )
    return embedder


@dataclass(frozen=True)
class RAGLiteConfig:
    """Configuration for RAGLite."""

    # LLM config used for generation.
    llm: Llama = field(default_factory=default_llm)
    llm_max_tries: int = 4
    llm_temperature: float = 1.0
    # Embedder config used for indexing.
    embedder: Llama = field(default_factory=default_embedder)
    embedder_batch_size: int = 128
    embedder_dtype: npt.DTypeLike = np.float16
    embedder_normalize: bool = True
    multi_vector_weight: float = 0.5  # Between 0 (chunk embedding) and 1 (sentence embedding).
    # Chunker config used to partition documents into chunks.
    chunk_max_size: int = 1440  # Max number of characters per chunk.
    chunk_sentence_window_size: int = 3
    # Database config.
    db_url: str | URL = "sqlite:///raglite.sqlite"
    # Vector search config.
    vector_search_index_id: str = "default"
    vector_search_index_metric: str = "cosine"  # The query adapter supports "dot" and "cosine".
