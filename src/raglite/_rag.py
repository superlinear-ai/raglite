"""Retrieval-augmented generation."""

import json
import logging
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from litellm import (  # type: ignore[attr-defined]
    ChatCompletionMessageToolCall,
    acompletion,
    completion,
    stream_chunk_builder,
    supports_function_calling,
)

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkSpan
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


def _count_tokens(item: str) -> int:
    """Estimate the number of tokens in an item."""
    return len(item) // 3


def _get_last_message_idx(messages: list[dict[str, str]], role: str) -> int | None:
    """Get the index of the last message with a specified role."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == role:
            return i
    return None


def _calculate_buffer_tokens(
    messages: list[dict[str, str]] | None,
    roles: list[str],
    user_prompt: str | None,
    template: str,
) -> int:
    """Calculate the number of tokens used by other messages."""
    # Calculate already used tokens (buffer)
    buffer = 0
    # Triggered when using tool calls
    if messages:
        # Count used tokens by the last message of each role
        for role in roles:
            idx = _get_last_message_idx(messages, role)
            if idx is not None:
                buffer += _count_tokens(json.dumps(messages[idx]))
        return buffer
    # Triggered when using add_context
    if user_prompt:
        return _count_tokens(template.format(context="", user_prompt=user_prompt))
    return 0


def _cutoff_idx(token_counts: list[int], max_tokens: int, *, reverse: bool = False) -> int:
    """Find the cutoff index in token counts to fit within max tokens."""
    counts = token_counts[::-1] if reverse else token_counts
    cum_tokens = np.cumsum(counts)
    cutoff_idx = int(np.searchsorted(cum_tokens, max_tokens, side="right"))
    return len(token_counts) - cutoff_idx if reverse else cutoff_idx


def _get_token_counts(items: Sequence[str | ChunkSpan | Mapping[str, str]]) -> list[int]:
    """Compute token counts for a list of items."""
    return [
        _count_tokens(item.to_xml())
        if isinstance(item, ChunkSpan)
        else _count_tokens(json.dumps(item, ensure_ascii=False))
        if isinstance(item, dict)
        else _count_tokens(item)
        if isinstance(item, str)
        else 0
        for item in items
    ]


def _limit_chunkspans(
    tool_chunk_spans: dict[str, list[ChunkSpan]],
    config: RAGLiteConfig,
    *,
    messages: list[dict[str, str]] | None = None,
    user_prompt: str | None = None,
    template: str = RAG_INSTRUCTION_TEMPLATE,
) -> dict[str, list[ChunkSpan]]:
    """Limit chunk spans to fit within the context window."""
    # Calculate already used tokens (buffer)
    buffer = _calculate_buffer_tokens(
        messages, ["user", "system", "assistant"], user_prompt, template
    )
    # Determine max tokens available for context
    max_tokens = get_context_size(config) - buffer
    # Compute token counts for all chunk spans per tool
    tool_tokens_list: dict[str, list[int]] = {}
    tool_total_tokens: dict[str, int] = {}
    total_tokens = 0
    for tool_id, chunk_spans in tool_chunk_spans.items():
        tokens_list = _get_token_counts(chunk_spans)
        tool_tokens_list[tool_id] = tokens_list
        tool_total = sum(tokens_list)
        tool_total_tokens[tool_id] = tool_total
        total_tokens += tool_total
    # Early exit if we're already under the limit
    if total_tokens <= max_tokens:
        return tool_chunk_spans
    # Allocate tokens proportionally and truncate
    total_chunk_spans = sum(len(spans) for spans in tool_chunk_spans.values())
    limited_tool_chunk_spans: dict[str, list[ChunkSpan]] = {}
    for tool_id, chunk_spans in tool_chunk_spans.items():
        if not chunk_spans:
            limited_tool_chunk_spans[tool_id] = []
            continue
        # Proportional allocation
        tool_max_tokens = max_tokens * tool_total_tokens[tool_id] // total_tokens
        # Find cutoff point
        cutoff_idx = _cutoff_idx(tool_tokens_list[tool_id], tool_max_tokens)
        limited_tool_chunk_spans[tool_id] = chunk_spans[
            :cutoff_idx
        ]  # Keep only up to cutoff (ChunkSpans are ordered in descending relevance)
    # Log warning if chunks were dropped
    new_total_chunk_spans = sum(len(spans) for spans in limited_tool_chunk_spans.values())
    if new_total_chunk_spans < total_chunk_spans:
        logger.warning(
            "RAG context was limited to %d out of %d chunks due to context window size. "
            "Consider using a model with a bigger context window or reducing the number of retrieved chunks.",
            new_total_chunk_spans,
            total_chunk_spans,
        )
    return limited_tool_chunk_spans


def add_context(
    user_prompt: str,
    context: list[ChunkSpan],
    config: RAGLiteConfig,
    *,
    rag_instruction_template: str = RAG_INSTRUCTION_TEMPLATE,
) -> dict[str, str]:
    """Convert a user prompt to a RAG instruction.

    The RAG instruction's format follows Anthropic's best practices [1].

    [1] https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
    """
    # Limit context to fit within the context window.
    limited_context = _limit_chunkspans({"temp": context}, config, user_prompt=user_prompt)["temp"]
    message = {
        "role": "user",
        "content": rag_instruction_template.format(
            context="\n".join(
                chunk_span.to_xml(index=i + 1) for i, chunk_span in enumerate(limited_context)
            ),
            user_prompt=user_prompt.strip(),
        ),
    }
    return message


def _clip(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """Left clip a messages array to avoid hitting the context limit."""
    token_counts = _get_token_counts(messages)
    cutoff_idx = _cutoff_idx(token_counts, max_tokens, reverse=True)
    idx_user = _get_last_message_idx(messages, "user")
    if cutoff_idx == len(messages) or (idx_user is not None and idx_user < cutoff_idx):
        warnings.warn(
            (
                f"Context window of {max_tokens} tokens exceeded."
                "Consider using a model with a bigger context window or reducing the number of retrieved chunks."
            ),
            stacklevel=2,
        )
        # Try to include both last system and user messages if they fit together.
        # If not, include just user if it fits, else return empty.
        idx_system = _get_last_message_idx(messages, "system")
        if (
            idx_user is not None
            and idx_system is not None
            and idx_system < idx_user
            and token_counts[idx_user] + token_counts[idx_system] <= max_tokens
        ):
            return [messages[idx_system], messages[idx_user]]
        if idx_user is not None and token_counts[idx_user] <= max_tokens:
            return [messages[idx_user]]
        return []
    return messages[cutoff_idx:]


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
    *,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, Any]]:
    """Run tools to search the knowledge base for RAG context."""
    tool_chunk_spans: dict[str, list[ChunkSpan]] = {}
    tool_messages: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if tool_call.function.name == "search_knowledge_base":
            kwargs = json.loads(tool_call.function.arguments)
            kwargs["config"] = config
            tool_chunk_spans[tool_call.id] = retrieve_context(**kwargs)
        else:
            error_message = f"Unknown function `{tool_call.function.name}`."
            raise ValueError(error_message)
    tool_chunk_spans = _limit_chunkspans(tool_chunk_spans, config, messages=messages)
    for tool_id, chunk_spans in tool_chunk_spans.items():
        tool_messages.append(
            {
                "role": "tool",
                "content": '{{"documents": [{elements}]}}'.format(
                    elements=", ".join(
                        chunk_span.to_json(index=i + 1) for i, chunk_span in enumerate(chunk_spans)
                    )
                ),
                "tool_call_id": tool_id,
            }
        )
        if chunk_spans and callable(on_retrieval):
            on_retrieval(chunk_spans)
    return tool_messages


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
        messages.extend(_run_tools(tool_calls, on_retrieval, config, messages=messages))
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
        messages.extend(_run_tools(tool_calls, on_retrieval, config, messages=messages))
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
