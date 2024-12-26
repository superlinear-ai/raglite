"""MCP server for RAGLite."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from raglite._config import RAGLiteConfig
from raglite._rag import create_rag_instruction, retrieve_rag_context

Query = Annotated[
    str,
    Field(
        description=(
            "The `query` string to search the knowledge base with.\n"
            "The `query` string MUST be a precise single-faceted question in the user's language.\n"
            "The `query` string MUST resolve all pronouns to explicit nouns from the conversation history."
        )
    ),
]


def create_mcp_server(server_name: str, *, config: RAGLiteConfig) -> FastMCP:
    """Create a RAGLite MCP server."""
    mcp = FastMCP(server_name)

    @mcp.prompt()
    def kb(query: Query) -> str:
        """Answer a question with information from the knowledge base."""
        chunk_spans = retrieve_rag_context(query, config=config)
        rag_instruction = create_rag_instruction(query, chunk_spans)
        return rag_instruction["content"]

    @mcp.tool()
    def search_knowledge_base(query: Query) -> str:
        """Search the knowledge base."""
        chunk_spans = retrieve_rag_context(query, config=config)
        rag_context = '{{"documents": [{elements}]}}'.format(
            elements=", ".join(
                chunk_span.to_json(index=i + 1) for i, chunk_span in enumerate(chunk_spans)
            )
        )
        return rag_context

    # Warm up the querying pipeline.
    if str(config.db_url).startswith("sqlite") or config.embedder.startswith("llama-cpp-python"):
        _ = retrieve_rag_context("Hello world", config=config)

    return mcp
