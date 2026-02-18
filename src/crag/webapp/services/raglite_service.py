from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any


def run_raglite(
    query: str,
    *,
    query_time: str,
    task: str,
    num_chunks: int,
    db_url: str,
    embedder: str,
    llm: str,
    self_query: bool = True,
    rerank: bool = False,
    hybrid_search: bool = False,
    agentic_rag: bool = False,
    capture_logs: bool = False,
    serialize_chunks: bool = False,
) -> dict[str, Any]:
    from crag.models.raglite import RAGLiteModel
    from raglite import add_context, rag

    model = RAGLiteModel(
        task=task,
        db_url=db_url or None,
        embedder=embedder or None,
        llm=llm or None,
        use_self_query=self_query,
        use_rerank=rerank,
        use_hybrid_search=hybrid_search,
        use_agentic_rag=agentic_rag,
    )
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    def _execute() -> tuple[list[Any], str]:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information from a collection of documents.\n"
                f"Today's date is {query_time}.",
            }
        ]

        chunk_spans: list[Any] = []
        if model.use_agentic_rag:
            messages.append({"role": "user", "content": query})
        else:
            if model.use_rerank:
                chunk_spans = model.get_chunks_via_rerank(query=query, num_chunks=num_chunks)
            else:
                from raglite import retrieve_context

                chunk_spans = retrieve_context(query=query, num_chunks=num_chunks, config=model.config)

            messages.append(add_context(user_prompt=query, context=chunk_spans, config=model.config))

        answer = ""
        stream = rag(
            messages,
            config=model.config,
            on_retrieval=chunk_spans.extend if model.use_agentic_rag else None,
        )
        for update in stream:
            answer += update
        return chunk_spans, answer

    if capture_logs:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            chunk_spans, answer = _execute()
    else:
        chunk_spans, answer = _execute()

    chunks: list[Any]
    if serialize_chunks:
        chunks = [_serialize_chunk_span(chunk_span) for chunk_span in chunk_spans]
    else:
        chunks = chunk_spans

    return {
        "self_query": self_query,
        "rerank": rerank,
        "hybrid_search": hybrid_search,
        "agentic_rag": agentic_rag,
        "answer": answer,
        "chunks": chunks,
        "stdout": stdout_capture.getvalue() if capture_logs else "",
        "stderr": stderr_capture.getvalue() if capture_logs else "",
    }


def _serialize_chunk_span(chunk_span: Any) -> dict[str, Any]:
    source = getattr(chunk_span, "document", None)
    filename = str(getattr(source, "filename", "") or "").strip()
    url = str(getattr(source, "url", "") or "").strip()
    metadata = getattr(source, "metadata_", None) if source else None
    content = str(getattr(chunk_span, "content", "") or "").strip()
    return {
        "filename": filename,
        "url": url,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "content": content,
    }
