"""Retrieval-augmented generation."""

from collections.abc import Iterator

from litellm import completion, get_model_info  # type: ignore[attr-defined]

from raglite._config import RAGLiteConfig
from raglite._database import Chunk
from raglite._litellm import LlamaCppPythonLLM
from raglite._search import hybrid_search, rerank, retrieve_segments
from raglite._typing import SearchMethod


def rag(
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: SearchMethod | list[str] | list[Chunk] = hybrid_search,
    config: RAGLiteConfig | None = None,
) -> Iterator[str]:
    """Retrieval-augmented generation."""
    # If the user has configured a llama-cpp-python model, we ensure that LiteLLM's model info is up
    # to date by loading that LLM.
    config = config or RAGLiteConfig()
    if config.llm.startswith("llama-cpp-python"):
        _ = LlamaCppPythonLLM.llm(config.llm)
    # Reduce the maximum number of contexts to take into account the LLM's context size.
    llm_provider = "llama-cpp-python" if config.llm.startswith("llama-cpp") else None
    model_info = get_model_info(config.llm, custom_llm_provider=llm_provider)
    max_tokens = (model_info.get("max_tokens") or 2048) - 256
    max_tokens_per_context = round(1.2 * (config.chunk_max_size // 4))
    max_tokens_per_context *= 1 + len(context_neighbors or [])
    max_contexts = min(max_contexts, max_tokens // max_tokens_per_context)
    # Retrieve the top chunks.
    chunks: list[str] | list[Chunk]
    if callable(search):
        # If the user has configured a reranker, we retrieve extra contexts to rerank.
        extra_contexts = 4 * max_contexts if config.reranker else 0
        # Retrieve relevant contexts.
        chunk_ids, _ = search(prompt, num_results=max_contexts + extra_contexts, config=config)
        # Rerank the relevant contexts.
        chunks = rerank(query=prompt, chunk_ids=chunk_ids, config=config)
    else:
        # The user has passed a list of chunk_ids or chunks directly.
        chunks = search
    # Extend the top contexts with their neighbors and group chunks into contiguous segments.
    segments = retrieve_segments(chunks[:max_contexts], neighbors=context_neighbors, config=config)
    # Respond with an LLM.
    contexts = "\n\n".join(
        f'<context index="{i}">\n{segment.strip()}\n</context>'
        for i, segment in enumerate(segments)
    )
    system_prompt = f"""
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.

{contexts}""".strip()
    stream = completion(
        model=config.llm,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    # Stream the response.
    for output in stream:
        token: str = output["choices"][0]["delta"].get("content") or ""
        yield token
