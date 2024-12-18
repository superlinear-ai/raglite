"""Chainlit frontend for RAGLite."""

import os
from functools import partial
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch, TextInput
from litellm.types.utils import ChatCompletionMessageToolCall

from raglite import (
    RAGLiteConfig,
    hybrid_search,
    insert_document,
    rerank_chunks,
    retrieve_rag_context,
)
from raglite._markdown import document_to_markdown
from raglite._rag import compose_rag_messages, rag, search_knowledge_base_tool
from raglite._tools import Tool

async_insert_document = cl.make_async(insert_document)


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat."""
    # Disable tokenizes parallelism to avoid the deadlock warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Add Chainlit settings with which the user can configure the RAGLite config.
    default_config = RAGLiteConfig()
    config = RAGLiteConfig(
        db_url=os.environ.get("RAGLITE_DB_URL", default_config.db_url),
        llm=os.environ.get("RAGLITE_LLM", default_config.llm),
        embedder=os.environ.get("RAGLITE_EMBEDDER", default_config.embedder),
    )
    settings = await cl.ChatSettings(  # type: ignore[no-untyped-call]
        [
            TextInput(id="db_url", label="Database URL", initial=str(config.db_url)),
            TextInput(id="llm", label="LLM", initial=config.llm),
            TextInput(id="embedder", label="Embedder", initial=config.embedder),
            Switch(id="vector_search_query_adapter", label="Query adapter", initial=True),
        ]
    ).send()
    await update_config(settings)


@cl.on_settings_update  # type: ignore[arg-type]
async def update_config(settings: cl.ChatSettings) -> None:
    """Update the RAGLite config."""
    # Update the RAGLite config given the Chainlit settings.
    config = RAGLiteConfig(
        db_url=settings["db_url"],  # type: ignore[index]
        llm=settings["llm"],  # type: ignore[index]
        embedder=settings["embedder"],  # type: ignore[index]
        vector_search_query_adapter=settings["vector_search_query_adapter"],  # type: ignore[index]
    )
    cl.user_session.set("config", config)  # type: ignore[no-untyped-call]
    # Run a search to prime the pipeline if it's a local pipeline.
    # TODO: Don't do this for SQLite once we switch from PyNNDescent to sqlite-vec.
    if str(config.db_url).startswith("sqlite") or config.embedder.startswith("llama-cpp-python"):
        # async with cl.Step(name="initialize", type="retrieval"):
        query = "Hello world"
        # TODO: Compare with previous commit (async and rerank)
        hybrid_search(query=query, config=config)


@cl.on_message
async def handle_message(user_message: cl.Message) -> None:
    """Respond to a user message."""
    # Get the config and message history from the user session.
    config: RAGLiteConfig = cl.user_session.get("config")  # type: ignore[no-untyped-call]
    # Determine what to do with the attachments.
    inline_attachments = []
    for file in user_message.elements:
        if file.path:
            doc_md = document_to_markdown(Path(file.path))
            if len(doc_md) // 3 <= 5 * (config.chunk_max_size // 3):
                # Document is small enough to attach to the context.
                inline_attachments.append(f"{Path(file.path).name}:\n\n{doc_md}")
            else:
                # Document is too large and must be inserted into the database.
                async with cl.Step(name="insert", type="run") as step:
                    step.input = Path(file.path).name
                    await async_insert_document(Path(file.path), config=config)
    # Append any inline attachments to the user prompt.
    user_prompt = (
        "\n\n".join(
            f'<attachment index="{i}">\n{attachment.strip()}\n</attachment>'
            for i, attachment in enumerate(inline_attachments)
        )
        + f"\n\n{user_message.content}"
    )

    # # Stream the LLM response.
    assistant_message = cl.Message(content="")
    messages = compose_rag_messages(
        user_prompt=user_prompt,
        history=cl.chat_context.to_openai()[:-1],  # type: ignore[no-untyped-call]
        system_prompt=None,
    )

    for token in rag(
        messages,
        config=config,
        tools=get_tools(config),
    ):
        await assistant_message.stream_token(token)

    await assistant_message.update()  # type: ignore[no-untyped-call]


@cl.step
def tool_progress(tool_call: ChatCompletionMessageToolCall) -> None:
    """Show the tool call."""
    step = cl.context.current_step
    step.input = tool_call.to_json()
    step.update()


def get_tools(config: RAGLiteConfig) -> list[Tool]:
    """Get the tools for the RAGLite pipeline."""
    return [
        Tool(
            name="weather",
            description="Get the weather at a location.",
            parameters={
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for.",
                }
            },
            call=cl.step(lambda location: f"The weather in {location} is sunny."),
        ),
        Tool(
            name="time",
            description="Get the current time.",
            parameters={
                "location": {
                    "type": "string",
                    "description": "The location to get the time for.",
                }
            },
            call=cl.step(
                lambda location: f"The current time in {location} is 12:00 PM.", type="tool"
            ),
        ),
        search_knowledge_base_tool(
            cl.step(
                partial(
                    retrieve_rag_context,
                    search_method=cl.step(hybrid_search, type="retrieval"),
                    rerank=cl.step(rerank_chunks, type="rerank"),
                    config=config,
                ),
                name="Search knowledge base",
                type="tool",
            )
        ),
    ]
