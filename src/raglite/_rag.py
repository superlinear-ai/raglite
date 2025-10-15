"""Retrieval-augmented generation."""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, ClassVar, Literal

import numpy as np
from litellm import (  # type: ignore[attr-defined]
    ChatCompletionMessageToolCall,
    acompletion,
    completion,
    stream_chunk_builder,
    supports_function_calling,
)
from pydantic import BaseModel, create_model

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkSpan
from raglite._extract import extract_with_llm
from raglite._insert import _get_database_metadata
from raglite._litellm import get_context_size
from raglite._search import retrieve_chunk_spans
from raglite._typing import MetadataFilter

logger = logging.getLogger(__name__)
# The default RAG instruction template follows Anthropic's best practices [1].
# [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
RAG_INSTRUCTION_TEMPLATE = """
---
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Whenever possible, use only the provided context to respond to the question at the end.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
---

<context>
{context}
</context>

{user_prompt}
""".strip()

SELF_QUERY_PROMPT = """
You extract metadata filters from user queries to help search a knowledge base.

Rules:
- Only populate a field when the query explicitly and unambiguously mentions a specific allowed value for that field
- If the query is general, ambiguous, or doesn't mention a field, leave it as None
- Do not infer values from common knowledge, popularity, or context from other fields
""".strip()


def retrieve_context(
    query: str,
    *,
    num_chunks: int = 10,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Retrieve context for RAG."""
    # Call the search method.
    config = config or RAGLiteConfig()
    # If self_query is enabled, extract metadata filters from the query.
    if config.self_query:
        self_query_filter = _self_query(query, config=config)
        metadata_filter = {**self_query_filter, **(metadata_filter or {})}
    results = config.search_method(
        query, num_results=num_chunks, metadata_filter=metadata_filter, config=config
    )
    # Convert results to chunk spans.
    chunk_spans = []
    if isinstance(results, tuple):
        chunk_spans = retrieve_chunk_spans(results[0], config=config)
    elif all(isinstance(result, Chunk) for result in results):
        chunk_spans = retrieve_chunk_spans(results, config=config)  # type: ignore[arg-type]
    elif all(isinstance(result, ChunkSpan) for result in results):
        chunk_spans = results  # type: ignore[assignment]
    return chunk_spans


def add_context(
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
            context="\n".join(
                chunk_span.to_xml(index=i + 1) for i, chunk_span in enumerate(context)
            ),
            user_prompt=user_prompt.strip(),
        ),
    }
    return message


def _clip(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """Left clip a messages array to avoid hitting the context limit."""
    cum_tokens = np.cumsum([len(message.get("content") or "") // 3 for message in messages][::-1])
    first_message = -np.searchsorted(cum_tokens, max_tokens)
    return messages[first_message:]


def _get_tools(
    messages: list[dict[str, str]], config: RAGLiteConfig
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | str | None]:
    """Get tools to search the knowledge base if no RAG context is provided in the messages."""
    # Check if messages already contain RAG context or if the LLM supports tool use.
    final_message = messages[-1].get("content", "")
    messages_contain_rag_context = any(
        s in final_message for s in ("<context>", "<document>", "from_chunk_id")
    )
    llm_supports_function_calling = supports_function_calling(config.llm)
    if not messages_contain_rag_context and not llm_supports_function_calling:
        error_message = "You must either explicitly provide RAG context in the last message, or use an LLM that supports function calling."
        raise ValueError(error_message)
    # Return a single tool to search the knowledge base if no RAG context is provided.
    tools: list[dict[str, Any]] | None = (
        [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "Search the knowledge base.\n"
                        "IMPORTANT: You MAY NOT use this function if the question can be answered with common knowledge or straightforward reasoning.\n"
                        "For multi-faceted questions, call this function once for each facet."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The `query` string MUST be a precise single-faceted question in the user's language.\n"
                                    "The `query` string MUST resolve all pronouns to explicit nouns."
                                ),
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        if not messages_contain_rag_context
        else None
    )
    tool_choice: dict[str, Any] | str | None = "auto" if tools else None
    return tools, tool_choice


def _run_tools(
    tool_calls: list[ChatCompletionMessageToolCall],
    on_retrieval: Callable[[list[ChunkSpan]], None] | None,
    config: RAGLiteConfig,
) -> list[dict[str, Any]]:
    """Run tools to search the knowledge base for RAG context."""
    tool_messages: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if tool_call.function.name == "search_knowledge_base":
            kwargs = json.loads(tool_call.function.arguments)
            kwargs["config"] = config
            chunk_spans = retrieve_context(**kwargs)
            tool_messages.append(
                {
                    "role": "tool",
                    "content": '{{"documents": [{elements}]}}'.format(
                        elements=", ".join(
                            chunk_span.to_json(index=i + 1)
                            for i, chunk_span in enumerate(chunk_spans)
                        )
                    ),
                    "tool_call_id": tool_call.id,
                }
            )
            if chunk_spans and callable(on_retrieval):
                on_retrieval(chunk_spans)
        else:
            error_message = f"Unknown function `{tool_call.function.name}`."
            raise ValueError(error_message)
    return tool_messages


def _self_query(
    query: str,
    *,
    system_prompt: str = SELF_QUERY_PROMPT,
    config: RAGLiteConfig | None = None,
) -> MetadataFilter:
    """Extract metadata filters from a natural language query."""
    config = config or RAGLiteConfig()
    # Retrieve the available metadata from the database.
    metadata_records = _get_database_metadata(config=config)
    if not metadata_records:
        return {}
    # Create dynamic Pydantic model for the metadata filter
    field_definitions: dict[str, Any] = {}
    field_definitions["system_prompt"] = (ClassVar[str], system_prompt)
    for record in metadata_records:
        field_definitions[record.name] = (Literal[tuple(record.values)] | None, None)
    metadata_filter_model = create_model(
        "MetadataFilterModel", **field_definitions, __base__=BaseModel
    )
    # Call extract_with_llm
    try:
        result = extract_with_llm(
            return_type=metadata_filter_model,
            user_prompt=query,
            config=config,
            temperature=0,
        )
    except ValueError as e:
        logger.debug("Failed to extract metadata filter: %s", e)
        return {}
    else:
        metadata_filter = result.model_dump()
        return {k: v for k, v in metadata_filter.items() if v is not None}


def rag(
    messages: list[dict[str, str]],
    *,
    on_retrieval: Callable[[list[ChunkSpan]], None] | None = None,
    config: RAGLiteConfig,
) -> Iterator[str]:
    # If the final message does not contain RAG context, get a tool to search the knowledge base.
    max_tokens = get_context_size(config)
    tools, tool_choice = _get_tools(messages, config)
    # Stream the LLM response, which is either a tool call request or an assistant response.
    stream = completion(
        model=config.llm,
        messages=_clip(messages, max_tokens),
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        if isinstance(token := chunk.choices[0].delta.content, str):
            yield token
    # Check if there are tools to be called.
    response = stream_chunk_builder(chunks, messages)
    tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]
    if tool_calls:
        # Add the tool call request to the message array.
        messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
        # Run the tool calls to retrieve the RAG context and append the output to the message array.
        messages.extend(_run_tools(tool_calls, on_retrieval, config))
        # Stream the assistant response.
        chunks = []
        stream = completion(model=config.llm, messages=_clip(messages, max_tokens), stream=True)
        for chunk in stream:
            chunks.append(chunk)
            if isinstance(token := chunk.choices[0].delta.content, str):
                yield token
    # Append the assistant response to the message array.
    response = stream_chunk_builder(chunks, messages)
    messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]


async def async_rag(
    messages: list[dict[str, str]],
    *,
    on_retrieval: Callable[[list[ChunkSpan]], None] | None = None,
    config: RAGLiteConfig,
) -> AsyncIterator[str]:
    # If the final message does not contain RAG context, get a tool to search the knowledge base.
    max_tokens = get_context_size(config)
    tools, tool_choice = _get_tools(messages, config)
    # Asynchronously stream the LLM response, which is either a tool call or an assistant response.
    async_stream = await acompletion(
        model=config.llm,
        messages=_clip(messages, max_tokens),
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
    )
    chunks = []
    async for chunk in async_stream:
        chunks.append(chunk)
        if isinstance(token := chunk.choices[0].delta.content, str):
            yield token
    # Check if there are tools to be called.
    response = stream_chunk_builder(chunks, messages)
    tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]
    if tool_calls:
        # Add the tool call requests to the message array.
        messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
        # Run the tool calls to retrieve the RAG context and append the output to the message array.
        # TODO: Make this async.
        messages.extend(_run_tools(tool_calls, on_retrieval, config))
        # Asynchronously stream the assistant response.
        chunks = []
        async_stream = await acompletion(
            model=config.llm, messages=_clip(messages, max_tokens), stream=True
        )
        async for chunk in async_stream:
            chunks.append(chunk)
            if isinstance(token := chunk.choices[0].delta.content, str):
                yield token
    # Append the assistant response to the message array.
    response = stream_chunk_builder(chunks, messages)
    messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
