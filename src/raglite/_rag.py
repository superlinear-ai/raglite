"""Retrieval-augmented generation."""

import json
import logging
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
The following context is intended to support answering the question below.
Use this information as the primary source when forming your response.
Provide a direct answer to the question without referencing how the information was supplied.
---

<context>
{context}
</context>

{user_prompt}
""".strip()

SEARCH_AGENT_PROMPT = """
You are an expert research assistant that helps retrieve the necessary information to answer a user's question.
You need to use a search tool that queries a knowledge base of documents. Each time you call the tool, you will receive a set of relevant document chunks as context.
You can use this context to iteratively refine your search and gather more information until you have enough to answer the user's question.
Once you do, respond with "Context is sufficient" and stop iterating.
*IMPORTANT*: You MUST iterate AS FEW TIMES AS POSSIBLE. Be strategic and efficient in your retrieval process.

## Query guidelines (tool calls)
- Each query must be a short, simple, precise question (single facet).
- Optimize questions for document retrieval: use keywords, explicit nouns/entities names, dates ...
- When a tool call does not return any relevant information, pivot your line of questioning.
Always consider prior asked questions before asking a new one:
    - DO NOT ask the same question twice.
    - DO NOT ask semantically overlapping questions.

## Example of bad questions:
- "What is the population of City A and City B?" (multi-faceted, not precise)
- "What is the population of City A?" followed by "What about City B?" (vague question, not optimized for retrieval)
- "What is the population of City A?" followed by "What is the population of City A?" (same question twice, not strategic)
- "When did David Gilmour join Pink Floyd and when did Syd Barrett leave? give months/years and reason" (multi-faceted, too complex)
- "Timeline of The Offspring band lineup changes drummers bassists guitarists with years (James Lilja, Ron Welty, Atom Willard, Pete Parada, Josh Freese, Brandon Pertzborn, Greg K., Todd Morse, Noodles)" (multi-faceted, too complex)
- "Has X ever had consecutive Billboard Hot 100 number-one singles? List any runs of consecutive Hot 100 #1 singles (song titles and dates) and the length of her longest such streak (as of August 26, 2024)." (multi-faceted, not precise, not optimized for retrieval)

## Example of good questions:
- "When did David Gilmour join Pink Floyd?" (single-faceted, precise)
- "Timeline of the Offspring" (optimized for retrieval, can be followed by more specific questions if needed)
- "What is the population of City A?" in parallel with "What is the population of City B?" (single-faceted, precise, non-repetitive)
- "When was X born?" instead of "How old is X?" (optimized for retrieval, as age depends on current date)
""".strip()

NO_TOOLS_FOLLOW_UP_PROMPT = """
Tools are unavailable for this step.
Do not call or reference any tool/function.
Try to answer the question to the best of your ability using only the context provided and your general knowledge.
If that is not possible, acknowledge it.
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
    if isinstance(results, tuple):
        return retrieve_chunk_spans(results[0], config=config)
    if all(isinstance(result, Chunk) for result in results):
        return retrieve_chunk_spans(results, config=config)  # type: ignore[arg-type]
    if all(isinstance(result, ChunkSpan) for result in results):
        return list(results)  # type: ignore[arg-type]
    return []


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
    user_prompt: str | None,
    template: str,
) -> int:
    """Calculate the number of tokens used by existing messages."""
    # Triggered when using tool calls: count all messages.
    if messages:
        return sum(_count_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
    # Triggered when using add_context: count template overhead.
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
    token_counts: list[int] = []
    for item in items:
        if isinstance(item, ChunkSpan):
            token_counts.append(_count_tokens(item.to_xml()))
        elif isinstance(item, Mapping):
            token_counts.append(_count_tokens(json.dumps(item, ensure_ascii=False)))
        else:
            token_counts.append(_count_tokens(item))
    return token_counts


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
    buffer = _calculate_buffer_tokens(messages, user_prompt, template)
    # Determine max tokens available for context, reserving space for the LLM's response.
    max_output_tokens = min(2048, get_context_size(config) // 4)
    max_tokens = get_context_size(config) - buffer - max_output_tokens
    # Compute token counts for all chunk spans per tool
    tool_tokens_list: dict[str, list[int]] = {}
    tool_total_tokens: dict[str, int] = {}
    total_tokens = 0
    total_chunk_spans = 0
    for tool_id, chunk_spans in tool_chunk_spans.items():
        tokens_list = _get_token_counts(chunk_spans)
        tool_tokens_list[tool_id] = tokens_list
        tool_total = sum(tokens_list)
        tool_total_tokens[tool_id] = tool_total
        total_tokens += tool_total
        total_chunk_spans += len(chunk_spans)
    # Early exit if we're already under the limit
    if total_tokens == 0 or total_tokens <= max_tokens:
        return tool_chunk_spans
    # Allocate tokens proportionally and truncate
    new_total_chunk_spans = 0
    scale_ratio = max_tokens / total_tokens
    limited_tool_chunk_spans: dict[str, list[ChunkSpan]] = {}
    for tool_id, chunk_spans in tool_chunk_spans.items():
        if not chunk_spans:
            limited_tool_chunk_spans[tool_id] = []
            continue
        # Proportional allocation
        tool_max_tokens = int(scale_ratio * tool_total_tokens[tool_id])
        # Find cutoff point
        cutoff_idx = _cutoff_idx(tool_tokens_list[tool_id], tool_max_tokens)
        limited_tool_chunk_spans[tool_id] = chunk_spans[
            :cutoff_idx
        ]  # Keep only up to cutoff (ChunkSpans are ordered in descending relevance)
        new_total_chunk_spans += cutoff_idx
    # Log warning if chunks were dropped
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
        logger.warning(
            "Context window of %d tokens exceeded. "
            "Consider using a model with a bigger context window or reducing the number of retrieved chunks.",
            max_tokens,
        )
        # Try to include both last system and user messages if they fit together.
        # If not, always preserve at least the last user message — the token estimate
        # is approximate, and dropping all messages guarantees a crash.
        idx_system = _get_last_message_idx(messages, "system")
        if (
            idx_user is not None
            and idx_system is not None
            and idx_system < idx_user
            and token_counts[idx_user] + token_counts[idx_system] <= max_tokens
        ):
            return [messages[idx_system], messages[idx_user]]
        if idx_user is not None:
            return [messages[idx_user]]
        return messages[-1:]
    return messages[cutoff_idx:]


def _get_tools(
    messages: list[dict[str, str]], config: RAGLiteConfig
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | str | None]:
    """Get tools to search the knowledge base if no RAG context is provided in the messages."""
    # Check if messages already contain RAG context or if the LLM supports tool use.
    final_message = messages[-1].get("content") or ""
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
                        "Search the knowledge base for contextual information needed to answer a user question.\n"
                        "IMPORTANT: You MAY NOT use this function if the question can be answered with common knowledge or straightforward reasoning.\n"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The exact user question, only rephrase if necessary for clarity. Add current date information if relevant.",
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


def _run_tool(
    tool_call: ChatCompletionMessageToolCall,
    config: RAGLiteConfig,
    *,
    metadata_filter: MetadataFilter | None = None,
) -> tuple[str, list[ChunkSpan]]:
    """
    Run a single tool to search the knowledge base.

    Returns the tool_id and the raw chunk_spans (before formatting/limiting).
    """
    if tool_call.function.name == "search_knowledge_base":
        try:
            query = json.loads(tool_call.function.arguments)["query"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            msg = f"Invalid arguments for 'search_knowledge_base': {exc}"
            raise ValueError(msg) from exc
        messages = [
            {
                "role": "system",
                "content": SEARCH_AGENT_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        tool = {
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": (
                    "Search the knowledge base with a single faceted question. "
                    "Multi-faceted questions are not allowed and should be broken down into multiple calls. \n"
                    "Example of a bad question: 'What is the population of City A and the GDP of Country B?'\n"
                    "Example of good questions: 'What is the population of City A?', 'What is the GDP of Country B?'\n"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A short, precise, single-faceted question.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

        # Start iterating and keep only chunk spans that introduce at least one new chunk ID.
        chunk_spans: list[ChunkSpan] = []
        seen_chunk_ids: set[str] = set()
        context_size = get_context_size(config)
        max_output_tokens = min(2048, context_size // 4)
        max_input_tokens = context_size - max_output_tokens
        for iteration_index in range(max(1, config.agentic_iterations)):
            response = completion(
                model=config.llm,
                messages=_clip(messages, max_input_tokens),
                tools=[tool],
                tool_choice="required" if iteration_index == 0 else "auto",
            )
            messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
            tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]

            # check if the tool call is valid
            if tool_calls:
                retrieved_chunk_spans: list[ChunkSpan] = []
                messages.extend(
                    _run_tools(
                        tool_calls,
                        retrieved_chunk_spans.extend,
                        config,
                        messages=messages,
                        metadata_filter=metadata_filter,
                    )
                )
                # Keep a span if it contains at least one chunk we have not seen before.
                novel_chunk_spans = [
                    chunk_span
                    for chunk_span in retrieved_chunk_spans
                    if any(chunk.id not in seen_chunk_ids for chunk in chunk_span.chunks)
                ]
                chunk_spans.extend(novel_chunk_spans)
                for chunk_span in novel_chunk_spans:
                    seen_chunk_ids.update(chunk.id for chunk in chunk_span.chunks)
            else:
                break

        # Return ID and data so the main function can aggregate and limit them
        return tool_call.id, chunk_spans

    if tool_call.function.name == "query_knowledge_base":
        kwargs = json.loads(tool_call.function.arguments)
        kwargs["config"] = config
        if metadata_filter is not None:
            kwargs["metadata_filter"] = metadata_filter
        chunk_spans = retrieve_context(**kwargs)
        # Return ID and data so the main function can aggregate and limit them
        return tool_call.id, chunk_spans
    error_message = f"Unknown function {tool_call.function.name}."
    raise ValueError(error_message)


def _run_tools(
    tool_calls: list[ChatCompletionMessageToolCall],
    on_retrieval: Callable[[list[ChunkSpan]], None] | None,
    config: RAGLiteConfig,
    *,
    messages: list[dict[str, str]] | None,
    metadata_filter: MetadataFilter | None = None,
) -> list[dict[str, Any]]:
    """Run tools in parallel, limit the total context, then format messages."""
    tool_chunk_spans: dict[str, list[ChunkSpan]] = {}

    # 1. Parallel Execution
    # We use the _run_tool helper to fetch data concurrently
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_run_tool, tool_call, config, metadata_filter=metadata_filter)
            for tool_call in tool_calls
        ]

        # Collect results as they finish
        try:
            for future in as_completed(futures):
                tool_id, spans = future.result()
                tool_chunk_spans[tool_id] = spans
        except Exception as e:
            executor.shutdown(cancel_futures=True)
            error_message = f"Error executing tool: {e}"
            raise ValueError(error_message) from e

    # 2. Limit Context (Global limiting across all tools)
    tool_chunk_spans = _limit_chunkspans(tool_chunk_spans, config, messages=messages)

    # 3. Formatting & Callbacks
    tool_messages: list[dict[str, Any]] = []

    # Iterate over the original tool_calls list to maintain the correct order
    for tool_call in tool_calls:
        tool_id = tool_call.id
        chunk_spans = tool_chunk_spans.get(tool_id, [])

        # Create the final message structure
        documents = ", ".join(
            chunk_span.to_json(index=i + 1) for i, chunk_span in enumerate(chunk_spans)
        )
        tool_messages.append(
            {
                "role": "tool",
                "content": f'{{"documents": [{documents}]}}',
                "tool_call_id": tool_id,
            }
        )

        # Trigger callback now that the spans are final and limited
        if chunk_spans and callable(on_retrieval):
            on_retrieval(chunk_spans)

    return tool_messages


def _stream_rag_response(
    messages: list[dict[str, str]], config: RAGLiteConfig, *, use_tools: bool = True
) -> Generator[str, None, list[Any]]:
    """Stream the RAG response, which may include tool calls for retrieval."""
    context_size = get_context_size(config)
    max_output_tokens = min(2048, context_size // 4)
    max_input_tokens = context_size - max_output_tokens
    tools, tool_choice = _get_tools(messages, config) if use_tools else (None, None)
    local_chunks: list[Any] = []
    stream = completion(
        model=config.llm,
        messages=_clip(messages, max_input_tokens),
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
        max_tokens=max_output_tokens,
    )
    for chunk in stream:
        local_chunks.append(chunk)
        if isinstance(token := chunk.choices[0].delta.content, str):  # type: ignore[union-attr]
            yield token
    return local_chunks


async def _async_stream_rag_response(
    messages: list[dict[str, str]],
    config: RAGLiteConfig,
    response_chunks: list[Any],
    *,
    use_tools: bool = True,
) -> AsyncIterator[str]:
    """Async version of _stream_rag_response."""
    context_size = get_context_size(config)
    max_output_tokens = min(2048, context_size // 4)
    max_input_tokens = context_size - max_output_tokens
    tools, tool_choice = _get_tools(messages, config) if use_tools else (None, None)
    async_stream = await acompletion(
        model=config.llm,
        messages=_clip(messages, max_input_tokens),
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
        max_tokens=max_output_tokens,
    )
    async for chunk in async_stream:
        response_chunks.append(chunk)
        if isinstance(token := chunk.choices[0].delta.content, str):
            yield token


def rag(
    messages: list[dict[str, str]],
    *,
    on_retrieval: Callable[[list[ChunkSpan]], None] | None = None,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig,
) -> Iterator[str]:
    """Run retrieval-augmented generation with the given messages and config."""
    working = list(messages)
    chunks = yield from _stream_rag_response(working, config)
    response = stream_chunk_builder(chunks, working)
    working.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
    tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]

    if tool_calls:
        working.extend(
            _run_tools(
                tool_calls,
                on_retrieval,
                config,
                messages=working,
                metadata_filter=metadata_filter,
            )
        )
        follow_up_messages = [*working, {"role": "system", "content": NO_TOOLS_FOLLOW_UP_PROMPT}]
        chunks = yield from _stream_rag_response(follow_up_messages, config, use_tools=False)
        response = stream_chunk_builder(chunks, follow_up_messages)
        working.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]

    messages.extend(working[len(messages) :])


async def async_rag(
    messages: list[dict[str, str]],
    *,
    on_retrieval: Callable[[list[ChunkSpan]], None] | None = None,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig,
) -> AsyncIterator[str]:
    """Run retrieval-augmented generation with the given messages and config."""
    working = list(messages)
    chunks: list[Any] = []
    async for token in _async_stream_rag_response(working, config, chunks):
        yield token
    response = stream_chunk_builder(chunks, working)
    working.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
    tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]

    if tool_calls:
        working.extend(
            _run_tools(
                tool_calls,
                on_retrieval,
                config,
                messages=working,
                metadata_filter=metadata_filter,
            )
        )
        follow_up_messages = [*working, {"role": "system", "content": NO_TOOLS_FOLLOW_UP_PROMPT}]
        chunks = []
        async for token in _async_stream_rag_response(
            follow_up_messages, config, chunks, use_tools=False
        ):
            yield token
        response = stream_chunk_builder(chunks, follow_up_messages)
        working.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]

    messages.extend(working[len(messages) :])
