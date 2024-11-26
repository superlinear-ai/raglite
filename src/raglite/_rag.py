"""Retrieval-augmented generation."""

from collections.abc import AsyncIterator, Iterator
from typing import cast

from litellm import acompletion, completion

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ContextSegment
from raglite._litellm import get_context_size
from raglite._search import hybrid_search, rerank_chunks, retrieve_segments
from raglite._typing import SearchMethod

RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
""".strip()


def _max_contexts(
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    messages: list[dict[str, str]] | None = None,
    config: RAGLiteConfig | None = None,
) -> int:
    """Determine the maximum number of contexts for RAG."""
    # Get the model's context size.
    config = config or RAGLiteConfig()
    max_tokens = get_context_size(config)
    # Reduce the maximum number of contexts to take into account the LLM's context size.
    max_context_tokens = (
        max_tokens
        - sum(len(message["content"]) // 3 for message in messages or [])  # Previous messages.
        - len(RAG_SYSTEM_PROMPT) // 3  # System prompt.
        - len(prompt) // 3  # User prompt.
    )
    max_tokens_per_context = config.chunk_max_size // 3
    max_tokens_per_context *= 1 + len(context_neighbors or [])
    max_contexts = min(max_contexts, max_context_tokens // max_tokens_per_context)
    if max_contexts <= 0:
        error_message = "Not enough context tokens available for RAG."
        raise ValueError(error_message)
    return max_contexts


def context_segments(  # noqa: PLR0913
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: SearchMethod | list[str] | list[Chunk] = hybrid_search,
    messages: list[dict[str, str]] | None = None,
    config: RAGLiteConfig | None = None,
) -> list[ContextSegment]:
    """Retrieve contexts for RAG."""
    # Determine the maximum number of contexts.
    max_contexts = _max_contexts(
        prompt,
        max_contexts=max_contexts,
        context_neighbors=context_neighbors,
        messages=messages,
        config=config,
    )
    # Retrieve the top chunks.
    config = config or RAGLiteConfig()
    chunks: list[str] | list[Chunk]
    if callable(search):
        # If the user has configured a reranker, we retrieve extra contexts to rerank.
        extra_contexts = 3 * max_contexts if config.reranker else 0
        # Retrieve relevant contexts.
        chunk_ids, _ = search(prompt, num_results=max_contexts + extra_contexts, config=config)
        # Rerank the relevant contexts.
        chunks = rerank_chunks(query=prompt, chunk_ids=chunk_ids, config=config)
    else:
        # The user has passed a list of chunk_ids or chunks directly.
        chunks = search
    # Extend the top contexts with their neighbors and group chunks into contiguous segments.
    segments = retrieve_segments(chunks[:max_contexts], neighbors=context_neighbors, config=config)
    return segments


def rag(  # noqa: PLR0913
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: SearchMethod | list[str] | list[Chunk] | list[ContextSegment] = hybrid_search,
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    config: RAGLiteConfig | None = None,
) -> Iterator[str]:
    """Retrieval-augmented generation."""
    # Get the contexts for RAG as contiguous segments of chunks.
    config = config or RAGLiteConfig()
    segments: list[ContextSegment]
    if isinstance(search, list) and any(isinstance(segment, ContextSegment) for segment in search):
        segments = cast(list[ContextSegment], search)
    else:
        segments = context_segments(
            prompt,
            max_contexts=max_contexts,
            context_neighbors=context_neighbors,
            search=search,  # type: ignore[arg-type]
            config=config,
        )
    # Stream the LLM response.
    stream = completion(
        model=config.llm,
        messages=_compose_messages(
            prompt=prompt, system_prompt=system_prompt, messages=messages, segments=segments
        ),
        stream=True,
    )
    for output in stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token


async def async_rag(  # noqa: PLR0913
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: SearchMethod | list[str] | list[Chunk] | list[ContextSegment] = hybrid_search,
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    config: RAGLiteConfig | None = None,
) -> AsyncIterator[str]:
    """Retrieval-augmented generation."""
    # Get the contexts for RAG as contiguous segments of chunks.
    config = config or RAGLiteConfig()
    segments: list[ContextSegment]
    if isinstance(search, list) and any(isinstance(segment, ContextSegment) for segment in search):
        segments = cast(list[ContextSegment], search)
    else:
        segments = context_segments(
            prompt,
            max_contexts=max_contexts,
            context_neighbors=context_neighbors,
            search=search,  # type: ignore[arg-type]
            config=config,
        )
    messages = _compose_messages(
        prompt=prompt, system_prompt=system_prompt, messages=messages, segments=segments
    )
    # Stream the LLM response.
    async_stream = await acompletion(model=config.llm, messages=messages, stream=True)
    async for output in async_stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token


def _compose_messages(
    prompt: str,
    system_prompt: str,
    messages: list[dict[str, str]] | None,
    segments: list[ContextSegment] | None,
) -> list[dict[str, str]]:
    """Compose the messages for the LLM, placing the context in the user position."""
    # Using the format recommended by Anthropic for documents in RAG
    # (https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips#essential-tips-for-long-context-prompts
    if not segments:
        return [
            {"role": "system", "content": system_prompt},
            *(messages or []),
            {"role": "user", "content": prompt},
        ]

    context_content = "<documents>\n" + "\n".join(str(seg) for seg in segments) + "\n</documents>"

    return [
        {"role": "system", "content": system_prompt},
        *(messages or []),
        {"role": "user", "content": prompt + "\n\n" + context_content},
    ]
