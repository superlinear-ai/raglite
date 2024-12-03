"""RAGLite."""

from raglite._cli import cli
from raglite._config import RAGLiteConfig
from raglite._eval import answer_evals, evaluate, insert_evals
from raglite._insert import insert_document
from raglite._query_adapter import update_query_adapter
from raglite._rag import async_rag, create_rag_instruction, rag, retrieve_rag_context
from raglite._search import (
    hybrid_search,
    keyword_search,
    rerank_chunks,
    retrieve_chunk_spans,
    retrieve_chunks,
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
    "retrieve_chunk_spans",
    "rerank_chunks",
    # RAG
    "retrieve_rag_context",
    "create_rag_instruction",
    "async_rag",
    "rag",
    # Query adapter
    "update_query_adapter",
    # Evaluate
    "insert_evals",
    "answer_evals",
    "evaluate",
    # CLI
    "cli",
]
