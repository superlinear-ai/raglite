"""RAGLite."""

from raglite._config import RAGLiteConfig
from raglite._eval import answer_evals, evaluate, insert_evals
from raglite._index import insert_document, update_vector_index
from raglite._query_adapter import update_query_adapter
from raglite._rag import rag
from raglite._search import (
    fusion_search,
    hybrid_search,
    keyword_search,
    retrieve_segments,
    vector_search,
)

__all__ = [
    # Config
    "RAGLiteConfig",
    # Index
    "insert_document",
    "update_vector_index",
    # Search
    "fusion_search",
    "hybrid_search",
    "keyword_search",
    "vector_search",
    "retrieve_segments",
    # RAG
    "rag",
    # Query adapter
    "update_query_adapter",
    # Evaluate
    "insert_evals",
    "answer_evals",
    "evaluate",
]
