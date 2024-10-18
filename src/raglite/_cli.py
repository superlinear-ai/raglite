"""Chainlit frontend for RAGLite."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import click

from raglite._config import RAGLiteConfig
from raglite._insert import insert_document
from raglite._rag import rag
from raglite._search import hybrid_search, rerank, retrieve_chunks

# Reduce the Chainlit log level because it logs on import.
logging.getLogger("chainlit").setLevel(logging.WARNING)

try:
    import chainlit as cl
    from chainlit.cli import run_chainlit
    from chainlit.input_widget import Switch, TextInput
except ImportError:
    from collections.abc import Callable
    from typing import Any

    class ChainlitStub:
        @staticmethod
        def on_chat_start(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        @staticmethod
        def on_settings_update(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        @staticmethod
        def on_message(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        class ChatSettings: ...

        class Message: ...

    cl = ChainlitStub()  # type: ignore[assignment]
    run_chainlit = None  # type: ignore[assignment]


async def async_generator(sync_generator: Iterator[str]) -> AsyncGenerator[str, None]:
    """Convert a synchronous generator to an asynchronous generator."""
    for item in sync_generator:
        yield item
        await asyncio.sleep(0)  # Yield control to the event loop


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat."""
    # Add Chainlit settings with which the user can configure the RAGLite config.
    config = RAGLiteConfig(
        db_url=os.environ["RAGLITE_DB_URL"],
        llm=os.environ["RAGLITE_LLM"],
        embedder=os.environ["RAGLITE_EMBEDDER"],
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


@click.group()
def cli() -> None:
    """RAGLite CLI."""


@cli.command()
@click.option("--db_url", type=str, help="Database URL", default=RAGLiteConfig().db_url)
@click.option("--llm", type=str, help="LiteLLM LLM", default=RAGLiteConfig().llm)
@click.option("--embedder", type=str, help="LiteLLM embedder", default=RAGLiteConfig().embedder)
def chainlit(db_url: str, llm: str, embedder: str) -> None:
    """Serve a Chainlit frontend with RAGLite."""
    os.environ["RAGLITE_DB_URL"] = os.environ.get("RAGLITE_DB_URL", db_url)
    os.environ["RAGLITE_LLM"] = os.environ.get("RAGLITE_LLM", llm)
    os.environ["RAGLITE_EMBEDDER"] = os.environ.get("RAGLITE_EMBEDDER", embedder)
    if run_chainlit is None:
        error_message = "To serve a Chainlit frontend, please install the `chainlit` extra."  # type: ignore[unreachable]
        raise ImportError(error_message)
    run_chainlit(__file__)
