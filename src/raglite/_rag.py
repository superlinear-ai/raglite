"""Retrieval-augmented generation."""

import json
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from litellm import (  # type: ignore[attr-defined]
    ChatCompletionMessageToolCall,
    acompletion,
    completion,
)

from raglite._database import ChunkSpan
from raglite._litellm import get_context_size
from raglite._prompts import DEFAULT_RAG_INSTRUCTION_TEMPLATE
from raglite._search import retrieve_chunk_spans, retrieve_chunks
from raglite._tools import Tool, process_stream

if TYPE_CHECKING:
    from raglite._config import RAGLiteConfig
    from raglite._typing import ChunkRerankingMethod, ChunkSearchMethod


def retrieve_rag_context(  # noqa: PLR0913
    query: str,
    *,
    search_method: "ChunkSearchMethod",
    rerank: Optional["ChunkRerankingMethod"] = None,
    max_chunk_spans: int | None = None,
    chunk_neighbors: tuple[int, ...] = (-1, 1),
    config: "RAGLiteConfig",
) -> list[ChunkSpan]:
    """Retrieve context for RAG."""
    chunk_ids, _ = search_method(query, config=config)
    # Rerank the chunks from most to least relevant.
    if rerank:
        chunks = rerank(query, chunk_ids=chunk_ids, config=config)
    else:
        chunks = retrieve_chunks(chunk_ids, config=config)
    context = retrieve_chunk_spans(chunks, chunk_neighbors=chunk_neighbors, config=config)
    return context[:max_chunk_spans]


def create_rag_instruction(
    user_prompt: str,
    context: list[ChunkSpan],
    *,
    rag_instruction_template: str = DEFAULT_RAG_INSTRUCTION_TEMPLATE,
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
            )
            if context
            else user_prompt,
        ),
    }
    return message


def compose_rag_messages(
    user_prompt: str,
    *,
    context: list[ChunkSpan] | None = None,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    rag_instruction_template: str | None = None,
) -> list[dict[str, str]]:
    """Compose a list of messages to generate a response."""
    messages = [
        *([{"role": "system", "content": system_prompt}] if system_prompt else []),
        *(history or []),
        create_rag_instruction(
            user_prompt=user_prompt,
            context=context if context else [],
            rag_instruction_template=rag_instruction_template or DEFAULT_RAG_INSTRUCTION_TEMPLATE,
        ),
    ]
    return messages


def _clip(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """Left clip a messages array to avoid hitting the context limit."""
    cum_tokens = np.cumsum([len(message.get("content") or "") // 3 for message in messages][::-1])
    first_message = -np.searchsorted(cum_tokens, max_tokens)
    return messages[first_message:]


def search_knowledge_base_tool(chunk_span_search_method: Callable[[str], list[ChunkSpan]]) -> Tool:
    def retrieval_tool(query: str) -> str:
        response = json.dumps([span.to_dict() for span in chunk_span_search_method(query)])
        return response

    return Tool(
        name="search_knowledge_base",
        description="Search the knowledge base.",
        parameters={
            "query": {
                "type": "string",
                "description": (
                    "The `query` string to search the knowledge base with.\n"
                    "The `query` string MUST satisfy ALL of the following criteria:\n"
                    "- The `query` string MUST be a precise question in the user's language.\n"
                    "- The `query` string MUST resolve all pronouns to explicit nouns from the conversation history."
                ),
            },
        },
        call=retrieval_tool,
    )


def rag(
    messages: list[dict[str, str]],
    *,
    tools: list[Tool] | None = None,
    on_tool_call: Callable[[ChatCompletionMessageToolCall], Any] | None = None,
    max_iterations: int = 10,
    config: "RAGLiteConfig",
) -> Iterator[str]:
    """Generate a response. Mutates the messages array in place."""
    # If the final message does not contain RAG context, get a tool to search the knowledge base.
    max_tokens = get_context_size(config)
    for _ in range(max_iterations):
        clipped_messages = _clip(messages, max_tokens)
        stream = completion(
            model=config.llm,
            messages=clipped_messages,
            tools=[tool.to_json() for tool in tools] if tools else None,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        yield from process_stream(stream, messages=messages, tools=tools, on_tool_call=on_tool_call)
        if messages[-1]["role"] != "tool":
            break


async def async_rag(
    messages: list[dict[str, str]],
    *,
    tools: list[Tool] | None = None,
    on_tool_call: Callable[[ChatCompletionMessageToolCall], Any] | None = None,
    max_iterations: int = 10,
    config: "RAGLiteConfig",
) -> AsyncGenerator[str, None]:
    """Generate a response. Mutates the messages array in place."""
    max_tokens = get_context_size(config)
    for _ in range(max_iterations):
        clipped_messages = _clip(messages, max_tokens)
        stream = acompletion(
            model=config.llm,
            messages=clipped_messages,
            tools=[tool.to_json() for tool in tools] if tools else None,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        for token in process_stream(
            stream,
            messages=messages,
            tools=tools,
            on_tool_call=on_tool_call,
        ):
            yield token
        if messages[-1]["role"] != "tool":
            break
