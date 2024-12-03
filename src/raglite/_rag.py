"""Retrieval-augmented generation."""

from collections.abc import AsyncIterator, Iterator

import numpy as np
from litellm import acompletion, completion

from raglite._config import RAGLiteConfig
from raglite._database import ChunkSpan
from raglite._litellm import get_context_size
from raglite._search import hybrid_search, rerank_chunks, retrieve_chunk_spans
from raglite._typing import SearchMethod

# The default RAG instruction template follows Anthropic's best practices [1].
# [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
RAG_INSTRUCTION_TEMPLATE = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Whenever possible, use only the provided context to respond to the question at the end.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.

{context}

{user_prompt}
""".strip()


def retrieve_rag_context(
    query: str,
    *,
    num_chunks: int = 5,
    chunk_neighbors: tuple[int, ...] | None = (-1, 1),
    search: SearchMethod = hybrid_search,
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Retrieve context for RAG."""
    # If the user has configured a reranker, we retrieve extra contexts to rerank.
    config = config or RAGLiteConfig()
    extra_chunks = 3 * num_chunks if config.reranker else 0
    # Search for relevant chunks.
    chunk_ids, _ = search(query, num_results=num_chunks + extra_chunks, config=config)
    # Rerank the chunks from most to least relevant.
    chunks = rerank_chunks(query, chunk_ids=chunk_ids, config=config)
    # Extend the top contexts with their neighbors and group chunks into contiguous segments.
    context = retrieve_chunk_spans(chunks[:num_chunks], neighbors=chunk_neighbors, config=config)
    return context


def create_rag_instruction(
    user_prompt: str,
    context: list[ChunkSpan],
    *,
    rag_instruction_template: str = RAG_INSTRUCTION_TEMPLATE,
) -> dict[str, str]:
    """Convert a user prompt to a RAG instruction.

    The RAG instruction's format follows Anthropic's best practices [1].

    [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
    """
    message = {
        "role": "user",
        "content": rag_instruction_template.format(
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
