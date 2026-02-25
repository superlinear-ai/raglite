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
You are an expert research assistant that answers the user's question using a search tool over a knowledge base (RAG).
You may perform up to {allowed_iterations} iterations. In each iteration, you may issue up to {max_questions_per_iteration} search tool queries.

Your job is to:
1) If the question can be answered via common knowledge or straightforward reasoning, answer directly without using the search tool.
2) If not, decide what information is required to answer the question.
3) Call the search tool with precise, single-faceted questions to retrieve that information from the knowledge base.
4) Produce a final answer grounded in retrieved evidence.

## Workflow (repeat until done or iterations exhausted)
At the start of each iteration:
- Briefly state what is currently missing (the minimum unknowns that block a confident answer).
- Generate search queries that directly target those unknowns.

After each retrieval:
- Extract the relevant facts (and ignore irrelevant text).
- Assess sufficiency:
    - SUFFICIENT if you can answer every part of the user question with direct support from the retrieved info.
    - INSUFFICIENT if any required fact is missing, ambiguous, or unsupported.
- If sufficient, stop searching and answer the question.
- If insufficient, call the tool again with queries that target ONLY what's missing.

## Query guidelines (tool calls)
- Use the tool ONLY when the answer is not common knowledge or requires knowledge-base-specific facts.
- Each tool call must be a single, precise question (single facet). Split multi-facet needs into separate calls.
- Resolve pronouns and vague references into explicit nouns/entities.
- Avoid redundancy:
    - Do not ask the same question twice.
    - Do not ask semantically overlapping questions unless you are disambiguating conflicting info.
- Prefer queries that request:
    - Definitions / canonical records first (IDs, names, dates).
    - Then relationships / comparisons.
    - Then edge cases / exceptions if needed.

## Termination and fallback
- Stop early if sufficient.
- If you hit iteration limits and still lack key facts:
    - Provide the best partial answer supported by evidence.
    - List missing information clearly.

## Example
Original user question: "Which city has a larger population, City A or City B?"
Reasoning: We need the population of both cities. This is not common knowledge, so we will use the search tool to find this information.
Iteration 1:
    - Tool Call 1: "Population of City A"
    - Tool Call 2: "Population of City B"
Retrieved information:
    - "The population of City A is 1,000,000."
    - "The population of the urban area of City B is 1,200,000"
Assessment: INSUFFICIENT (urban area population includes city population and surroundings, so we cannot confidently compare)
Iteration 2:
    - Tool Call 1: "Population of City B (city proper)"
Retrieved information:
    - "The population of the city proper of City B is 900,000."
Assessment: SUFFICIENT (now we have comparable population figures for both cities)
Final answer: "City A has a larger population than City B. City A has a population of 1,000,000, while City B has a population of 900,000."
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
    return [
        (
            _count_tokens(item.to_xml())
            if isinstance(item, ChunkSpan)
            else (
                _count_tokens(json.dumps(item, ensure_ascii=False))
                if isinstance(item, dict)
                else _count_tokens(item)
                if isinstance(item, str)
                else 0
            )
        )
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
    if total_tokens <= max_tokens:
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
                        "Search the knowledge base using single-faceted questions.\n"
                        "For multi-faceted questions (comparison, sets, ..), call this function once for each facet.\n "
                        "Example original query: Which artist has the most number-one albums on the Billboard 200: X or Y?\n"
                        "Facet 1: How many albums on the Billboard 200 does X have? \n"
                        "Facet 2: How many albums on the Billboard 200 does Y have? \n"
                        "IMPORTANT: You MAY NOT use this function if the question can be answered with common knowledge or straightforward reasoning.\n"
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


def _run_tool(
    tool_call: ChatCompletionMessageToolCall,
    config: RAGLiteConfig,
) -> tuple[str, list[ChunkSpan]]:
    """
    Run a single tool to search the knowledge base.

    Returns the tool_id and the raw chunk_spans (before formatting/limiting).
    """
    if tool_call.function.name == "search_knowledge_base":
        kwargs = json.loads(tool_call.function.arguments)
        kwargs["config"] = config
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
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run tools in parallel, limit the total context, then format messages."""
    tool_chunk_spans: dict[str, list[ChunkSpan]] = {}

    # 1. Parallel Execution
    # We use the _run_tool helper to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_tool, tool_call, config) for tool_call in tool_calls]

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
    total_before = sum(len(spans) for spans in tool_chunk_spans.values())
    tool_chunk_spans = _limit_chunkspans(tool_chunk_spans, config, messages=messages)
    total_after = sum(len(spans) for spans in tool_chunk_spans.values())
    logger.info("Retrieved %d chunk span(s) across %d tool call(s) (%d after limiting).",
                total_before, len(tool_calls), total_after)

    # 3. Formatting & Callbacks
    tool_messages: list[dict[str, Any]] = []

    # Iterate over the original tool_calls list to maintain the correct order
    for tool_call in tool_calls:
        tool_id = tool_call.id
        chunk_spans = tool_chunk_spans.get(tool_id, [])

        # Create the final message structure
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

        # Trigger callback now that the spans are final and limited
        if chunk_spans and callable(on_retrieval):
            on_retrieval(chunk_spans)

    return tool_messages


def _stream_response(
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | str | None,
    context_size: int,
    config: RAGLiteConfig,
) -> Generator[str, None, Any]:
    """Stream an LLM response, yielding tokens and returning the assembled response."""
    max_output_tokens = min(2048, context_size // 4)
    max_input_tokens = context_size - max_output_tokens
    chunks: list[Any] = []
    stream = completion(
        model=config.llm,
        messages=_clip(messages, max_input_tokens),
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_output_tokens,
        stream=True,
    )
    for chunk in stream:
        chunks.append(chunk)
        if isinstance(token := chunk.choices[0].delta.content, str):  # type: ignore[union-attr]
            yield token
    return stream_chunk_builder(chunks, messages)


def rag(
    messages: list[dict[str, str]],
    *,
    on_retrieval: Callable[[list[ChunkSpan]], None] | None = None,
    allowed_iterations: int = 20,
    config: RAGLiteConfig,
) -> Iterator[str]:
    """Run retrieval-augmented generation with iterative tool calling."""
    assert allowed_iterations >= 1, "allowed_iterations must be at least 1"

    context_size = get_context_size(config)
    tools, tool_choice = _get_tools(messages, config)

    # Inject a system prompt to guide iterative retrieval in agentic mode.
    if tools:
        logger.info("Starting agentic RAG (up to %d iterations).", allowed_iterations)
        messages.insert(0, {
            "role": "system",
            "content": SEARCH_AGENT_PROMPT.format(
                allowed_iterations=allowed_iterations,
                max_questions_per_iteration=3,
            ),
        })

    # Stream the initial LLM response.
    response = yield from _stream_response(
        messages, tools=tools, tool_choice=tool_choice, context_size=context_size, config=config
    )

    # Iterative tool-calling loop: execute tool calls and stream follow-up responses.
    for iteration in range(allowed_iterations):
        tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]
        if not tool_calls:
            logger.info("Retrieval loop stopped after %d iteration(s): LLM returned no tool calls.", iteration)
            break

        queries = [
            json.loads(tc.function.arguments).get("query", "")
            for tc in tool_calls
            if tc.function.name == "search_knowledge_base"
        ]
        logger.info("Iteration %d: %d tool call(s) — queries: %s", iteration + 1, len(tool_calls), queries)

        messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
        messages.extend(_run_tools(tool_calls, on_retrieval, config, messages=messages))

        # On the final allowed iteration, withhold tools to force a direct answer.
        is_final = iteration == allowed_iterations - 1
        if is_final:
            logger.info("Iteration limit reached (%d). Forcing final answer.", allowed_iterations)
        response = yield from _stream_response(
            messages,
            tools=None if is_final else tools,
            tool_choice=None if is_final else tool_choice,
            context_size=context_size,
            config=config,
        )
    else:
        logger.info("Retrieval loop exhausted all %d iterations.", allowed_iterations)

    # Remove the injected system prompt before returning.
    if tools:
        messages.pop(0)

    # Append the final assistant response to the message array.
    messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]


async def async_rag(
    messages: list[dict[str, str]],
    *,
    on_retrieval: Callable[[list[ChunkSpan]], None] | None = None,
    allowed_iterations: int = 20,
    config: RAGLiteConfig,
) -> AsyncIterator[str]:
    """Async retrieval-augmented generation with iterative tool calling."""
    assert allowed_iterations >= 1, "allowed_iterations must be at least 1"

    context_size = get_context_size(config)
    max_output_tokens = min(2048, context_size // 4)
    max_input_tokens = context_size - max_output_tokens
    tools, tool_choice = _get_tools(messages, config)

    # Inject a system prompt to guide iterative retrieval in agentic mode.
    if tools:
        logger.info("Starting async agentic RAG (up to %d iterations).", allowed_iterations)
        messages.insert(0, {
            "role": "system",
            "content": SEARCH_AGENT_PROMPT.format(
                allowed_iterations=allowed_iterations,
                max_questions_per_iteration=3,
            ),
        })

    response: Any = None

    async def _async_stream(
        current_tools: list[dict[str, Any]] | None,
        current_tool_choice: dict[str, Any] | str | None,
    ) -> AsyncIterator[str]:
        nonlocal response
        chunks: list[Any] = []
        async_stream = await acompletion(
            model=config.llm,
            messages=_clip(messages, max_input_tokens),
            tools=current_tools,
            tool_choice=current_tool_choice,
            max_tokens=max_output_tokens,
            stream=True,
        )
        async for chunk in async_stream:
            chunks.append(chunk)
            if isinstance(token := chunk.choices[0].delta.content, str):
                yield token
        response = stream_chunk_builder(chunks, messages)

    # Stream the initial LLM response.
    async for token in _async_stream(tools, tool_choice):
        yield token

    # Iterative tool-calling loop: execute tool calls and stream follow-up responses.
    for iteration in range(allowed_iterations):
        tool_calls = response.choices[0].message.tool_calls  # type: ignore[union-attr]
        if not tool_calls:
            logger.info("Retrieval loop stopped after %d iteration(s): LLM returned no tool calls.", iteration)
            break

        queries = [
            json.loads(tc.function.arguments).get("query", "")
            for tc in tool_calls
            if tc.function.name == "search_knowledge_base"
        ]
        logger.info("Iteration %d: %d tool call(s) — queries: %s", iteration + 1, len(tool_calls), queries)

        messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
        # TODO: Make _run_tools async for true async execution.
        messages.extend(_run_tools(tool_calls, on_retrieval, config, messages=messages))

        # On the final allowed iteration, withhold tools to force a direct answer.
        is_final = iteration == allowed_iterations - 1
        if is_final:
            logger.info("Iteration limit reached (%d). Forcing final answer.", allowed_iterations)
        async for token in _async_stream(
            None if is_final else tools,
            None if is_final else tool_choice,
        ):
            yield token
    else:
        logger.info("Retrieval loop exhausted all %d iterations.", allowed_iterations)

    # Remove the injected system prompt before returning.
    if tools:
        messages.pop(0)

    # Append the final assistant response to the message array.
    messages.append(response.choices[0].message.to_dict())  # type: ignore[arg-type,union-attr]
