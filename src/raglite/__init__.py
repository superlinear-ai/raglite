"""RAGLite."""

from raglite._cli import cli
from raglite._config import RAGLiteConfig
from raglite._eval import answer_evals, evaluate, insert_evals
from raglite._insert import insert_document
from raglite._query_adapter import update_query_adapter
from raglite._rag import async_generate, generate, get_context_segments
from raglite._search import (
    hybrid_search,
    keyword_search,
    rerank_chunks,
    retrieve_chunks,
    retrieve_segments,
    vector_search,
)

__all__ = [
    # Config
    "RAGLiteConfig",
    "answer_evals",
    "async_generate",
    # CLI
    "cli",
    "evaluate",
    # RAG
    "generate",
    "get_context_segments",
    # Search
    "hybrid_search",
    # Insert
    "insert_document",
    # Evaluate
    "insert_evals",
    "keyword_search",
    "rerank_chunks",
    "retrieve_chunks",
    "retrieve_segments",
    # Query adapter
    "update_query_adapter",
    "vector_search",
]
