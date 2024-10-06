"""RAGLite."""

from raglite._config import RAGLiteConfig
from raglite._eval import answer_evals, evaluate, insert_evals
from raglite._insert import insert_document
from raglite._query_adapter import update_query_adapter
from raglite._rag import rag
from raglite._search import (
    hybrid_search,
    keyword_search,
    rerank,
    retrieve_chunks,
    retrieve_segments,
    vector_search,
)

__all__ = [
    # Config
    "RAGLiteConfig",
    # Insert
    "insert_document",
    # Search
    "hybrid_search",
    "keyword_search",
    "vector_search",
    "retrieve_chunks",
    "retrieve_segments",
    "rerank",
    # RAG
    "rag",
    # Query adapter
    "update_query_adapter",
    # Evaluate
    "insert_evals",
    "answer_evals",
    "evaluate",
]
