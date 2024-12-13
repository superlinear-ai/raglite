"""Retrieval-augmented generation."""

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Optional

import numpy as np
from litellm import acompletion, completion

from raglite._database import ChunkSpan
from raglite._litellm import get_context_size
from raglite._search import retrieve_chunk_spans, retrieve_chunks

if TYPE_CHECKING:
    from raglite._config import RAGLiteConfig
    from raglite._typing import ChunkRerankingMethod, ChunkSearchMethod


def retrieve_rag_context(  # noqa: PLR0913
    query: str,
    *,
    search: "ChunkSearchMethod",
    rerank: Optional["ChunkRerankingMethod"] = None,
    max_chunk_spans: int | None = None,
    chunk_neighbors: tuple[int, ...] = (-1, 1),
    config: "RAGLiteConfig",
) -> list[ChunkSpan]:
    """Retrieve context for RAG."""
    chunk_ids, _ = search(query, config=config)
    # Rerank the chunks from most to least relevant.
    if rerank:
        chunks = rerank(query, chunk_ids=chunk_ids, config=config)
    else:
        chunks = retrieve_chunks(chunk_ids, config=config)
    context = retrieve_chunk_spans(chunks, chunk_neighbors=chunk_neighbors, config=config)
    return context[:max_chunk_spans]


def create_rag_instruction(
    user_prompt: str, context: list[ChunkSpan], *, config: "RAGLiteConfig"
) -> dict[str, str]:
    """Convert a user prompt to a RAG instruction.

    The RAG instruction's format follows Anthropic's best practices [1].

    [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
    """
    message = {
        "role": "user",
        "content": config.rag_instruction_template.format(
            user_prompt=user_prompt.strip(),
            context="\n".join(
                chunk_span.to_xml(index=i + 1) for i, chunk_span in enumerate(context)
            ),
        ),
    }
    return message


def rag(messages: list[dict[str, str]], *, config: "RAGLiteConfig") -> Iterator[str]:
    # Truncate the oldest messages so we don't hit the context limit.
    max_tokens = get_context_size(config)
    cum_tokens = np.cumsum([len(message.get("content", "")) // 3 for message in messages][::-1])
    messages = messages[-np.searchsorted(cum_tokens, max_tokens) :]
    # Stream the LLM response.
    stream = completion(model=config.llm, messages=messages, stream=True)
    for output in stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token


async def async_rag(
    messages: list[dict[str, str]], *, config: "RAGLiteConfig"
) -> AsyncIterator[str]:
    # Truncate the oldest messages so we don't hit the context limit.
    max_tokens = get_context_size(config)
    cum_tokens = np.cumsum([len(message.get("content", "")) // 3 for message in messages][::-1])
    messages = messages[-np.searchsorted(cum_tokens, max_tokens) :]
    # Asynchronously stream the LLM response.
    async_stream = await acompletion(model=config.llm, messages=messages, stream=True)
    async for output in async_stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token
