"""Retrieval-augmented generation."""

from collections.abc import AsyncIterator, Iterator

from litellm import acompletion, completion, get_model_info  # type: ignore[attr-defined]

from raglite._config import RAGLiteConfig
from raglite._database import Chunk
from raglite._litellm import LlamaCppPythonLLM
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
    # If the user has configured a llama-cpp-python model, we ensure that LiteLLM's model info is up
    # to date by loading that LLM.
    config = config or RAGLiteConfig()
    if config.llm.startswith("llama-cpp-python"):
        _ = LlamaCppPythonLLM.llm(config.llm)
    # Get the model's maximum context size.
    llm_provider = "llama-cpp-python" if config.llm.startswith("llama-cpp") else None
    model_info = get_model_info(config.llm, custom_llm_provider=llm_provider)
    max_tokens = model_info.get("max_tokens") or 2048
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


def _contexts(  # noqa: PLR0913
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: SearchMethod | list[str] | list[Chunk] = hybrid_search,
    messages: list[dict[str, str]] | None = None,
    config: RAGLiteConfig | None = None,
) -> list[str]:
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
    search: SearchMethod | list[str] | list[Chunk] = hybrid_search,
    messages: list[dict[str, str]] | None = None,
    config: RAGLiteConfig | None = None,
) -> Iterator[str]:
    """Retrieval-augmented generation."""
    # Get the contexts for RAG as contiguous segments of chunks.
    config = config or RAGLiteConfig()
    segments = _contexts(
        prompt,
        max_contexts=max_contexts,
        context_neighbors=context_neighbors,
        search=search,
        config=config,
    )
    system_prompt = f"{RAG_SYSTEM_PROMPT}\n\n" + "\n\n".join(
        f'<context index="{i}">\n{segment.strip()}\n</context>'
        for i, segment in enumerate(segments)
    )
    # Stream the LLM response.
    stream = completion(
        model=config.llm,
        messages=[
            *(messages or []),
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
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
    search: SearchMethod | list[str] | list[Chunk] = hybrid_search,
    messages: list[dict[str, str]] | None = None,
    config: RAGLiteConfig | None = None,
) -> AsyncIterator[str]:
    """Retrieval-augmented generation."""
    # Get the contexts for RAG as contiguous segments of chunks.
    config = config or RAGLiteConfig()
    segments = _contexts(
        prompt,
        max_contexts=max_contexts,
        context_neighbors=context_neighbors,
        search=search,
        config=config,
    )
    system_prompt = f"{RAG_SYSTEM_PROMPT}\n\n" + "\n\n".join(
        f'<context index="{i}">\n{segment.strip()}\n</context>'
        for i, segment in enumerate(segments)
    )
    # Stream the LLM response.
    async_stream = await acompletion(
        model=config.llm,
        messages=[
            *(messages or []),
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    async for output in async_stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token
