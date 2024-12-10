"""Retrieval-augmented generation."""

from collections.abc import AsyncIterator, Iterator
from dataclasses import replace

import numpy as np
from litellm import acompletion, completion

from raglite._config import RAGLiteConfig
from raglite._database import ChunkSpan
from raglite._litellm import get_context_size
from raglite._search import rerank_chunks, retrieve_chunk_spans, search


def retrieve_rag_context(query: str, *, config: RAGLiteConfig | None = None) -> list[ChunkSpan]:
    """Retrieve context for RAG."""
    # If the user has configured a reranker, we retrieve extra contexts to rerank.
    config = config or RAGLiteConfig()
    oversampled_num_chunks = (
        config.reranker_oversample * config.num_chunks if config.reranker else config.num_chunks
    )
    # Search for relevant chunks.
    chunk_ids, _ = search(query, config=replace(config, num_chunks=oversampled_num_chunks))
    # Rerank the chunks from most to least relevant.
    chunks = rerank_chunks(query, chunk_ids=chunk_ids, config=config)
    # Extend the top contexts with their neighbors and group chunks into contiguous segments.
    context = retrieve_chunk_spans(chunks[: config.num_chunks], config=config)
    return context


def create_rag_instruction(
    user_prompt: str, context: list[ChunkSpan], *, config: RAGLiteConfig | None = None
) -> dict[str, str]:
    """Convert a user prompt to a RAG instruction.

    The RAG instruction's format follows Anthropic's best practices [1].

    [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
    """
    config = config or RAGLiteConfig()
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


def rag(messages: list[dict[str, str]], *, config: RAGLiteConfig) -> Iterator[str]:
    # Truncate the oldest messages so we don't hit the context limit.
    max_tokens = get_context_size(config)
    cum_tokens = np.cumsum([len(message.get("content", "")) // 3 for message in messages][::-1])
    messages = messages[-np.searchsorted(cum_tokens, max_tokens) :]
    # Stream the LLM response.
    stream = completion(model=config.llm, messages=messages, stream=True)
    for output in stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token


async def async_rag(messages: list[dict[str, str]], *, config: RAGLiteConfig) -> AsyncIterator[str]:
    # Truncate the oldest messages so we don't hit the context limit.
    max_tokens = get_context_size(config)
    cum_tokens = np.cumsum([len(message.get("content", "")) // 3 for message in messages][::-1])
    messages = messages[-np.searchsorted(cum_tokens, max_tokens) :]
    # Asynchronously stream the LLM response.
    async_stream = await acompletion(model=config.llm, messages=messages, stream=True)
    async for output in async_stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token
