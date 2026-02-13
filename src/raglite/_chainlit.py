"""Chainlit frontend for RAGLite."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import chainlit as cl
from chainlit.input_widget import Switch, TextInput

from raglite import (
    Document,
    RAGLiteConfig,
    async_rag,
    hybrid_search,
    insert_documents,
    rerank_chunks,
)
from raglite._markdown import document_to_markdown

if TYPE_CHECKING:
    from raglite._database import ChunkSpan

async_insert_documents = cl.make_async(insert_documents)
async_hybrid_search = cl.make_async(hybrid_search)
async_rerank_chunks = cl.make_async(rerank_chunks)


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat."""
    # Set tokenizers parallelism to avoid a deadlock warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
    if config.embedder.startswith("llama-cpp-python"):
        query = "Hello world"
        chunk_ids, _ = await async_hybrid_search(query=query, config=config)
        _ = await async_rerank_chunks(query=query, chunk_ids=chunk_ids, config=config)


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
                    document = Document.from_path(Path(file.path))
                    await async_insert_documents([document], config=config)
    # Append any inline attachments to the user prompt.
    user_prompt = (
        "\n\n".join(
            f'<attachment index="{i}">\n{attachment.strip()}\n</attachment>'
            for i, attachment in enumerate(inline_attachments)
        )
        + f"\n\n{user_message.content}"
    ).strip()
    # Stream the LLM response.
    assistant_message = cl.Message(content="")
    chunk_spans: list[ChunkSpan] = []
    messages: list[dict[str, str]] = cl.chat_context.to_openai()[:-1]  # type: ignore[no-untyped-call]
    messages.append({"role": "user", "content": user_prompt})
    async for token in async_rag(messages, on_retrieval=chunk_spans.extend, config=config):
        await assistant_message.stream_token(token)
    # Append RAG sources, if any.
    if chunk_spans:
        rag_sources: dict[str, list[str]] = {}
        for chunk_span in chunk_spans:
            rag_sources.setdefault(chunk_span.document.id, [])
            rag_sources[chunk_span.document.id].append(str(chunk_span))
        assistant_message.content += "\n\nSources: " + ", ".join(  # Rendered as hyperlinks.
            f"[{i + 1}]" for i in range(len(rag_sources))
        )
        assistant_message.elements = [  # Markdown content is rendered in sidebar.
            cl.Text(name=f"[{i + 1}]", content="\n\n---\n\n".join(content), display="side")  # type: ignore[misc]
            for i, (_, content) in enumerate(rag_sources.items())
        ]
    await assistant_message.update()  # type: ignore[no-untyped-call]
