"""Retrieval-augmented generation."""

from collections.abc import Callable
from typing import Iterator

from llama_cpp import Llama
from sqlalchemy.engine import URL

from raglite.llm import default_llm
from raglite.search import get_chunks, hybrid_search

SYSTEM_PROMPT = """
Answer the user's question with the given contexts only.

{contexts}
""".strip()


def rag(  # noqa: PLR0913
    prompt: str,
    num_contexts: int = 5,
    context_neighbors: tuple[int, ...] = (-1, 1),
    temperature: float = 0.7,
    search: Callable[[str], tuple[list[int], list[float]]] = hybrid_search,
    llm: Callable[[], Llama] = default_llm,
    db_url: str | URL = "sqlite:///raglite.sqlite",
) -> Iterator[str]:
    """Retrieval-augmented generation."""
    # Retrieve relevant chunks.
    chunk_rowids, _ = search(prompt, num_results=num_contexts, db_url=db_url)
    chunks = get_chunks(chunk_rowids, neighbors=context_neighbors)
    # Convert the chunks into contexts.
    contexts = ""
    for i, chunk in enumerate(chunks):
        contexts += f"Context {i + 1}:\n```\n{chunk}\n```" + ("\n\n" if i < len(chunks) - 1 else "")
    # Respond with an LLM.
    system_prompt = SYSTEM_PROMPT.format(contexts=contexts)
    stream = llm().create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stream=True,
    )
    # Stream the response.
    for output in stream:
        yield output["choices"][0]["delta"].get("content", "")
