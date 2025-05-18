"""RAGLite config."""

import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from sqlalchemy.engine import URL

from raglite._lazy_llama import llama_supports_gpu_offload
from raglite._typing import ChunkId, SearchMethod

# Suppress rerankers output on import until [1] is fixed.
# [1] https://github.com/AnswerDotAI/rerankers/issues/36
with contextlib.redirect_stdout(StringIO()):
    from rerankers.models.flashrank_ranker import FlashRankRanker
    from rerankers.models.ranker import BaseRanker


cache_path = Path(user_data_dir("raglite", ensure_exists=True))


# Lazily load the default search method to avoid circular imports.
# TODO: Replace with search_and_rerank_chunk_spans after benchmarking.
def _vector_search(
    query: str, *, num_results: int = 10, config: "RAGLiteConfig | None" = None
) -> tuple[list[ChunkId], list[float]]:
    from raglite._search import vector_search

    return vector_search(query, num_results=num_results, config=config)


@dataclass(frozen=True)
class RAGLiteConfig:
    """RAGLite config."""

    # Database config.
    db_url: str | URL = f"duckdb:///{(cache_path / 'raglite.db').as_posix()}"
    # LLM config used for generation.
    llm: str = field(
        default_factory=lambda: (
            "llama-cpp-python/unsloth/Qwen3-8B-GGUF/*Q4_K_M.gguf@8192"
            if llama_supports_gpu_offload()
            else "llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192"
        )
    )
    llm_max_tries: int = 4
    # Embedder config used for indexing.
    embedder: str = field(
        default_factory=lambda: (  # Nomic-embed may be better if only English is used.
            "llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512"
            if llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 4  # noqa: PLR2004
            else "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@512"
        )
    )
    embedder_normalize: bool = True
    # Chunk config used to partition documents into chunks.
    chunk_max_size: int = 2048  # Max number of characters per chunk.
    # Vector search config.
    vector_search_index_metric: Literal["cosine", "dot", "l2"] = "cosine"
    vector_search_multivector: bool = True
    vector_search_query_adapter: bool = True  # Only supported for "cosine" and "dot" metrics.
    # Reranking config.
    reranker: BaseRanker | dict[str, BaseRanker] | None = field(
        default_factory=lambda: {
            "en": FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0, cache_dir=cache_path),
            "other": FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0, cache_dir=cache_path),
        },
        compare=False,  # Exclude the reranker from comparison to avoid lru_cache misses.
    )
    # Search config: you can pick any search method that returns (list[ChunkId], list[float]),
    # list[Chunk], or list[ChunkSpan].
    search_method: SearchMethod = field(default=_vector_search, compare=False)
