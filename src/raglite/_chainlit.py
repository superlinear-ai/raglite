"""Chainlit frontend for RAGLite."""

import asyncio
import os
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch, TextInput

from raglite import RAGLiteConfig, hybrid_search, insert_document, rag, rerank, retrieve_chunks


async def async_generator(sync_generator: Iterator[str]) -> AsyncGenerator[str, None]:
    """Convert a synchronous generator to an asynchronous generator."""
    for item in sync_generator:
        yield item
        await asyncio.sleep(0)  # Yield control to the event loop


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat."""
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
        async with cl.Step(name="initialize", type="retrieval"):
            query = "Hello world"
            chunk_ids, _ = await cl.make_async(hybrid_search)(query=query, config=config)
            _ = await cl.make_async(rerank)(query=query, chunk_ids=chunk_ids, config=config)


@cl.on_message
async def handle_message(user_message: cl.Message) -> None:
    """Respond to a user message."""
    # Get the config and message history from the user session.
    config: RAGLiteConfig = cl.user_session.get("config")  # type: ignore[no-untyped-call]
    # Insert any attached documents into the database.
    for file in user_message.elements:
        if file.path:
            async with cl.Step(name="insert", type="run") as step:
                step.input = Path(file.path).name
                await cl.make_async(insert_document)(Path(file.path), config=config)
    # Search for relevant contexts for RAG.
    async with cl.Step(name="search", type="retrieval") as step:
        step.input = user_message.content
        chunk_ids, _ = await cl.make_async(hybrid_search)(
            query=user_message.content, num_results=20, config=config
        )
        chunks = await cl.make_async(retrieve_chunks)(chunk_ids=chunk_ids, config=config)
        step.output = chunks
        step.elements = [  # Show the top 3 chunks inline.
            cl.Text(content=str(chunk), display="inline") for chunk in chunks[:3]
        ]
    # Rerank the chunks.
    async with cl.Step(name="rerank", type="rerank") as step:
        step.input = chunks
        chunks = await cl.make_async(rerank)(
            query=user_message.content, chunk_ids=chunks, config=config
        )
        step.output = chunks
        step.elements = [  # Show the top 3 chunks inline.
            cl.Text(content=str(chunk), display="inline") for chunk in chunks[:3]
        ]
    # Stream the RAG response.
    assistant_message = cl.Message(content="")
    stream = async_generator(
        rag(
            prompt=user_message.content,
            search=chunks,
            messages=cl.chat_context.to_openai()[-5:],  # type: ignore[no-untyped-call]
            config=config,
        )
    )
    async for token in stream:
        await assistant_message.stream_token(token)
    await assistant_message.update()  # type: ignore[no-untyped-call]
