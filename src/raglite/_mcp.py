"""MCP server for RAGLite."""

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from raglite._config import RAGLiteConfig
from raglite._rag import add_context, retrieve_context

Query = Annotated[
    str,
    Field(
        description=(
            "The `query` string MUST be a precise single-faceted question in the user's language.\n"
            "The `query` string MUST resolve all pronouns to explicit nouns."
        )
    ),
]


def create_mcp_server(server_name: str, *, config: RAGLiteConfig) -> FastMCP[Any]:
    """Create a RAGLite MCP server."""
    mcp: FastMCP[Any] = FastMCP(server_name)

    @mcp.prompt()
    def kb(query: Query) -> str:
        """Answer a question with information from the knowledge base."""
        chunk_spans = retrieve_context(query, config=config)
        rag_instruction = add_context(query, chunk_spans)
        return rag_instruction["content"]

    @mcp.tool()
    def search_knowledge_base(query: Query) -> str:
        """Search the knowledge base.

        IMPORTANT: You MAY NOT use this function if the question can be answered with common
        knowledge or straightforward reasoning. For multi-faceted questions, call this function once
        for each facet.
        """
        chunk_spans = retrieve_context(query, config=config)
        rag_context = '{{"documents": [{elements}]}}'.format(
            elements=", ".join(
                chunk_span.to_json(index=i + 1) for i, chunk_span in enumerate(chunk_spans)
            )
        )
        return rag_context

    # Warm up the querying pipeline.
    if config.embedder.startswith("llama-cpp-python"):
        _ = retrieve_context("Hello world", config=config)

    return mcp
