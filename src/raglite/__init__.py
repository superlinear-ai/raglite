"""RAGLite."""

from raglite._config import RAGLiteConfig
from raglite._eval import insert_evals
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
    "insert_document",
    "update_vector_index",
    "insert_evals",
    "update_query_adapter",
    "rag",
    "fusion_search",
    "hybrid_search",
    "keyword_search",
    "vector_search",
    "retrieve_segments",
    "RAGLiteConfig",
]
