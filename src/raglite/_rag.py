"""Retrieval-augmented generation."""

from collections.abc import Callable, Iterator

from raglite._config import RAGLiteConfig
from raglite._search import hybrid_search, retrieve_segments


def rag(
    prompt: str,
    *,
    num_contexts: int = 5,
    context_neighbors: tuple[int, ...] | None = (-1, 1),
    search: Callable[[str], tuple[list[int], list[float]]] = hybrid_search,
    config: RAGLiteConfig | None = None,
) -> Iterator[str]:
    """Retrieval-augmented generation."""
    # Retrieve relevant chunks.
    config = config or RAGLiteConfig()
    chunk_rowids, _ = search(prompt, num_results=num_contexts, config=config)
    chunks = retrieve_segments(chunk_rowids, neighbors=context_neighbors)
    # Respond with an LLM.
    contexts = "\n\n".join(
        f'<context index="{i}">\n{chunk.strip()}\n</context>' for i, chunk in enumerate(chunks)
    )
    system_prompt = f"""
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
Don't reference the context as if it were provided to you, nor as a document or text that can be referenced.
Instead, use the context as if it is part of your working memory.

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
        yield output["choices"][0]["delta"].get("content", "")
