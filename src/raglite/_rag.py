"""Retrieval-augmented generation."""

from collections.abc import Callable, Iterator

from raglite._config import RAGLiteConfig
from raglite._search import hybrid_search, retrieve_segments


def rag(
    prompt: str,
    *,
    max_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: Callable[[str], tuple[list[str], list[float]]] = hybrid_search,
    config: RAGLiteConfig | None = None,
) -> Iterator[str]:
    """Retrieval-augmented generation."""
    # Reduce the maximum number of contexts to take into account the LLM's context size.
    config = config or RAGLiteConfig()
    max_tokens = config.llm.n_ctx() - 256  # Account for the system and user prompts.
    max_tokens_per_context = round(1.2 * (config.chunk_max_size // 4))
    max_tokens_per_context *= 1 + len(context_neighbors or [])
    max_contexts = min(max_contexts, max_tokens // max_tokens_per_context)
    # Retrieve relevant contexts.
    chunk_ids, _ = search(prompt, num_results=max_contexts, config=config)  # type: ignore[call-arg]
    segments = retrieve_segments(chunk_ids, neighbors=context_neighbors)
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
    stream = config.llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=config.llm_temperature,
        stream=True,
    )
    # Stream the response.
    for output in stream:
        token: str = output["choices"][0]["delta"].get("content", "")  # type: ignore[assignment,index,union-attr]
        yield token
