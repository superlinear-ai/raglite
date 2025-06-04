"""RAGLite."""

from raglite._config import RAGLiteConfig
from raglite._database import Document
from raglite._eval import answer_evals, evaluate, insert_evals
from raglite._insert import insert_documents
from raglite._query_adapter import update_query_adapter
from raglite._rag import add_context, async_rag, rag, retrieve_context
from raglite._search import (
    hybrid_search,
    keyword_search,
    rerank_chunks,
    retrieve_chunk_spans,
    retrieve_chunks,
    search_and_rerank_chunk_spans,
    search_and_rerank_chunks,
    vector_search,
)

__all__ = [
    # Config
    "RAGLiteConfig",
    # Insert
    "Document",
    "insert_documents",
    # Search
    "hybrid_search",
    "keyword_search",
    "vector_search",
    "retrieve_chunks",
    "retrieve_chunk_spans",
    "rerank_chunks",
    "search_and_rerank_chunks",
    "search_and_rerank_chunk_spans",
    # RAG
    "retrieve_context",
    "add_context",
    "async_rag",
    "rag",
    # Query adapter
    "update_query_adapter",
    # Evaluate
    "answer_evals",
    "insert_evals",
    "evaluate",
]
