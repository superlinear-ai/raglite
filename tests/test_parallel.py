"""Exercise opt-in web retrieval through RAG with scripted model responses."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from mcp.client.streamable_http import streamablehttp_client

from raglite import ParallelSearchSession, RAGLiteConfig, async_rag, rag
from raglite._config import _vector_search


class ScriptedModel:
    """Choose a tool deterministically and consume its result in the response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.force_tool: str | None = None
        self.forge_metadata = False
        self.multiple_calls = False

    def complete(self, **kwargs: Any) -> Iterator[ModelResponseStream]:
        """Supply scripted chunks while retaining the real stream assembler."""
        self.requests.append(kwargs)
        if kwargs["tools"] is None:
            results = [
                message["content"] for message in kwargs["messages"] if message["role"] == "tool"
            ]
            answer = "Retrieved: " + " ".join(results) if results else "Answer without retrieval."
            yield ModelResponseStream(
                id="scripted-answer",
                model=kwargs["model"],
                choices=[
                    StreamingChoices(index=0, delta=Delta(content=answer), finish_reason="stop")
                ],
            )
            return
        user_prompt = next(
            message["content"]
            for message in reversed(kwargs["messages"])
            if message["role"] == "user"
        )
        tool_name = self.force_tool or (
            "parallel_web_fetch" if "fetch" in user_prompt else "parallel_web_search"
        )
        if not self.force_tool:
            assert tool_name in {tool["function"]["name"] for tool in kwargs["tools"]}
        arguments: dict[str, Any] = (
            {"urls": ["https://example.org/source"], "objective": "Find the release date"}
            if tool_name == "parallel_web_fetch"
            else {"objective": user_prompt, "search_queries": ["example latest release"]}
        )
        if self.forge_metadata:
            arguments.update(session_id="model-chosen-id", model_name="model-chosen-model")
        calls = [
            {
                "index": 0,
                "id": "web-call",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ]
        if self.multiple_calls:
            calls.append(
                {
                    "index": 1,
                    "id": "local-call",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": '{"query":"local facts"}',
                    },
                }
            )
        yield ModelResponseStream(
            id="scripted-tools",
            model=kwargs["model"],
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_validate({"tool_calls": calls}),
                    finish_reason="tool_calls",
                )
            ],
        )

    async def complete_async(self, **kwargs: Any) -> AsyncIterator[ModelResponseStream]:
        """Expose the same scripted choice through the async model interface."""

        async def chunks() -> AsyncIterator[ModelResponseStream]:
            for chunk in self.complete(**kwargs):
                yield chunk

        return chunks()


class MCPEndpoint:
    """A deterministic JSON-RPC HTTP endpoint behind the real MCP SDK."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.calls: list[dict[str, Any]] = []
        self.error: str | None = None
        self.result_text: str | None = None
        self.missing_fetch = False
        self.pause = False
        self.started: asyncio.Event | None = None
        self.cancelled = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        """Record the real destination and headers before returning MCP responses."""
        self.requests.append(request)
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert (
            request.headers["User-Agent"] == "RAGLite (+https://github.com/superlinear-ai/raglite)"
        )
        if request.method != "POST":
            return httpx.Response(405)
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "scripted-parallel", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": f"Scripted {name}",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "objective": {"type": "string"},
                                "search_queries": {"type": "array", "items": {"type": "string"}},
                                "urls": {"type": "array", "items": {"type": "string"}},
                                "session_id": {"type": "string"},
                                "model_name": {"type": "string"},
                            },
                            "required": ["objective"] if name == "web_search" else ["urls"],
                        },
                    }
                    for name in (
                        ("web_search",) if self.missing_fetch else ("web_search", "web_fetch")
                    )
                ]
            }
        elif method == "tools/call":
            self.calls.append(payload["params"])
            if self.pause:
                if self.started is not None:
                    self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": self.error
                        or self.result_text
                        or json.dumps(
                            {
                                "results": [
                                    {
                                        "url": "https://example.org/source",
                                        "title": "Release",
                                        "excerpts": ["The release date is 2026-10-15."],
                                    }
                                ],
                            }
                        ),
                    }
                ],
                "isError": self.error is not None,
            }
        else:
            pytest.fail(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def transport(self, url: str, **kwargs: Any) -> Any:
        """Keep the SDK HTTP transport and replace only network delivery."""

        def create_client(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.MockTransport(self.handle),
                headers=headers,
                timeout=timeout,
                auth=auth,
            )

        return streamablehttp_client(url, httpx_client_factory=create_client, **kwargs)


@pytest.fixture
def web_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig]:
    """Use production resolver, dispatcher, MCP codec, and result consumption."""
    endpoint = MCPEndpoint()
    model = ScriptedModel()
    monkeypatch.setattr("raglite._parallel.streamablehttp_client", endpoint.transport)
    monkeypatch.setattr("raglite._rag.completion", model.complete)
    monkeypatch.setattr("raglite._rag.acompletion", model.complete_async)
    monkeypatch.setattr("raglite._rag.get_context_size", lambda _: 16384)
    config = RAGLiteConfig(llm="gpt-4o-mini", embedder="text-embedding-3-small", reranker=None)
    return endpoint, model, config


def test_sync_rag_web_results_and_conversation_lifetime(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search, fetch, and follow-up use one identity and produce source-bearing answers."""
    endpoint, model, config = web_runtime
    monkeypatch.setenv("PARALLEL_API_KEY", "ambient-credential-must-not-be-used")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-model-key")
    session = ParallelSearchSession()
    messages = [{"role": "user", "content": "Find the latest release"}]
    for prompt in (None, "fetch the source page", "Find another release"):
        if prompt:
            messages.append({"role": "user", "content": prompt})
        answer = "".join(rag(messages, config=config, web_search=session))
        assert "2026-10-15" in answer
        assert "https://example.org/source" in answer
    assert [call["name"] for call in endpoint.calls] == ["web_search", "web_fetch", "web_search"]
    identifiers = {call["arguments"]["session_id"] for call in endpoint.calls}
    assert len(identifiers) == 1
    assert UUID(identifiers.pop()).version == 4  # noqa: PLR2004
    assert all(call["arguments"]["model_name"] == config.llm for call in endpoint.calls)
    assert all("Authorization" not in request.headers for request in endpoint.requests)
    methods = [
        json.loads(request.content).get("method")
        for request in endpoint.requests
        if request.method == "POST"
    ]
    assert methods.count("tools/list") >= 1
    assert methods.count("tools/call") == len(endpoint.calls)
    for request in model.requests:
        if request["tools"] is not None:
            for tool in request["tools"]:
                if tool["function"]["name"].startswith("parallel_"):
                    properties = tool["function"]["parameters"]["properties"]
                    assert "session_id" not in properties
                    assert "model_name" not in properties
    assert config.search_method is _vector_search


def test_independent_and_new_chats_get_distinct_ids(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """A shared config and transport do not share identities across chats or reset."""
    endpoint, _, config = web_runtime
    conversation_count = 3
    for _ in range(conversation_count):
        messages = [{"role": "user", "content": "Find the release"}]
        assert "2026-10-15" in "".join(
            rag(messages, config=config, web_search=ParallelSearchSession())
        )
    assert len({call["arguments"]["session_id"] for call in endpoint.calls}) == conversation_count


def test_async_concurrent_conversations_keep_their_ids(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """Interleave independent RAG conversations sharing configuration and transport."""
    endpoint, _, config = web_runtime

    async def conversation(label: str) -> None:
        session = ParallelSearchSession()
        messages = [{"role": "user", "content": f"Find release {label}"}]
        for _ in range(2):
            answer = "".join(
                [token async for token in async_rag(messages, config=config, web_search=session)]
            )
            assert "https://example.org/source" in answer
            messages.append({"role": "user", "content": f"Find another release {label}"})

    async def run() -> None:
        await asyncio.gather(conversation("alpha"), conversation("beta"))

    asyncio.run(run())
    by_conversation: dict[str, set[str]] = {"alpha": set(), "beta": set()}
    for call in endpoint.calls:
        arguments = call["arguments"]
        label = arguments["objective"].split()[-1]
        by_conversation[label].add(arguments["session_id"])
    assert all(len(identifiers) == 1 for identifiers in by_conversation.values())
    assert by_conversation["alpha"] != by_conversation["beta"]


def test_runtime_metadata_cannot_be_overridden_and_key_is_explicit(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """Model arguments cannot replace trusted identity/model or explicit credentials."""
    endpoint, model, config = web_runtime
    model.forge_metadata = True
    messages = [{"role": "user", "content": "Find a release"}]
    list(
        rag(messages, config=config, web_search=ParallelSearchSession(api_key="explicit-test-key"))
    )
    assert endpoint.calls[0]["arguments"]["session_id"] != "model-chosen-id"
    assert endpoint.calls[0]["arguments"]["model_name"] == config.llm
    assert all(
        request.headers["Authorization"] == "Bearer explicit-test-key"
        for request in endpoint.requests
    )


def test_disabled_web_has_no_network_or_web_tools(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unchanged local default exposes no web tools and makes no MCP requests."""
    endpoint, model, config = web_runtime
    model.force_tool = "search_knowledge_base"
    observed: list[Any] = []

    def local_tools(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        observed.append((args, kwargs))
        return [{"role": "tool", "tool_call_id": "web-call", "content": "Local document result"}]

    monkeypatch.setattr("raglite._rag._run_tools", local_tools)
    messages = [{"role": "user", "content": "Find local facts"}]
    answer = "".join(rag(messages, config=config, metadata_filter={"topic": ["Physics"]}))
    assert "Local document result" in answer
    assert not endpoint.requests
    assert [tool["function"]["name"] for tool in model.requests[0]["tools"]] == [
        "search_knowledge_base"
    ]
    assert observed[0][1]["metadata_filter"] == {"topic": ["Physics"]}
    assert observed[0][0][2] is config


def test_manual_context_suppresses_remote_discovery(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """Explicit RAG context keeps both web and local tools suppressed."""
    endpoint, model, config = web_runtime
    messages = [{"role": "user", "content": "<context>Private document.</context> Answer this."}]
    answer = "".join(rag(messages, config=config, web_search=ParallelSearchSession()))
    assert answer == "Answer without retrieval."
    assert not endpoint.requests
    assert model.requests[0]["tools"] is None


def test_mixed_local_and_web_calls_preserve_order_and_filters(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Web opt-in keeps the local dispatcher, saved config, and metadata filter."""
    endpoint, model, config = web_runtime
    model.multiple_calls = True
    observed: list[Any] = []

    def local_tools(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        observed.append((args, kwargs))
        return [{"role": "tool", "tool_call_id": "local-call", "content": "Local document result"}]

    monkeypatch.setattr("raglite._rag._run_tools", local_tools)
    messages = [{"role": "user", "content": "Find local and web release facts"}]
    answer = "".join(
        rag(
            messages,
            config=config,
            metadata_filter={"topic": ["Physics"]},
            web_search=ParallelSearchSession(),
        )
    )
    assert "Local document result" in answer
    assert "https://example.org/source" in answer
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        "web-call",
        "local-call",
    ]
    assert observed[0][1]["metadata_filter"] == {"topic": ["Physics"]}
    assert observed[0][0][2] is config
    assert len(endpoint.calls) == 1


def test_quota_error_retry_keeps_conversation_identity(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """A failed remote call propagates without fallback and retains retry identity."""
    endpoint, _, config = web_runtime
    session = ParallelSearchSession()
    messages = [{"role": "user", "content": "Find a release"}]
    endpoint.error = "Anonymous free quota exceeded; retry later."
    with pytest.raises(ValueError, match="quota exceeded"):
        list(rag(messages, config=config, web_search=session))
    assert messages == [{"role": "user", "content": "Find a release"}]
    endpoint.error = None
    assert "2026-10-15" in "".join(rag(messages, config=config, web_search=session))
    assert len(endpoint.calls) == 2  # noqa: PLR2004
    assert (
        endpoint.calls[0]["arguments"]["session_id"] == endpoint.calls[1]["arguments"]["session_id"]
    )


def test_async_timeout_cancels_mcp_request(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """The configured timeout covers the actual dispatched network request."""
    endpoint, _, config = web_runtime
    endpoint.pause = True

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            _ = [
                token
                async for token in async_rag(
                    [{"role": "user", "content": "Find release"}],
                    config=config,
                    web_search=ParallelSearchSession(timeout=0.1),
                )
            ]

    asyncio.run(run())
    assert endpoint.calls
    assert endpoint.cancelled


def test_async_cancellation_reaches_mcp_request(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """Cancelling a RAG caller closes its outstanding web request."""
    endpoint, _, config = web_runtime
    endpoint.pause = True

    async def run() -> None:
        endpoint.started = asyncio.Event()

        async def consume() -> None:
            _ = [
                token
                async for token in async_rag(
                    [{"role": "user", "content": "Find release"}],
                    config=config,
                    web_search=ParallelSearchSession(),
                )
            ]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(endpoint.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert endpoint.cancelled


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(timeout: float) -> None:
    """Reject invalid timeouts before enabling a remote session."""
    with pytest.raises(ValueError, match="positive finite"):
        ParallelSearchSession(timeout=timeout)


def test_unoffered_tool_is_rejected_before_dispatch(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """A model cannot promote a nested knowledge-base function to a public tool."""
    endpoint, model, config = web_runtime
    model.force_tool = "query_knowledge_base"
    with pytest.raises(ValueError, match="not offered"):
        list(
            rag(
                [{"role": "user", "content": "Find a release"}],
                config=config,
                web_search=ParallelSearchSession(),
            )
        )
    assert not endpoint.calls


def test_unsupported_model_rejects_web_access_before_discovery(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model without function calling cannot activate remote tools."""
    endpoint, _, config = web_runtime
    monkeypatch.setattr("raglite._rag.supports_function_calling", lambda _: False)
    with pytest.raises(ValueError, match="supports function calling"):
        list(
            rag(
                [{"role": "user", "content": "Find a release"}],
                config=config,
                web_search=ParallelSearchSession(),
            )
        )
    assert not endpoint.requests


def test_incomplete_discovery_fails_without_model_or_search_call(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
) -> None:
    """An incompatible endpoint cannot silently provide partial remote support."""
    endpoint, model, config = web_runtime
    endpoint.missing_fetch = True
    with pytest.raises(ValueError, match="did not provide"):
        list(
            rag(
                [{"role": "user", "content": "Find a release"}],
                config=config,
                web_search=ParallelSearchSession(),
            )
        )
    assert not endpoint.calls
    assert not model.requests


def test_web_excerpts_respect_context_budget_with_local_results(
    web_runtime: tuple[MCPEndpoint, ScriptedModel, RAGLiteConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large web excerpts leave room for local evidence and the final answer."""
    endpoint, model, config = web_runtime
    model.multiple_calls = True
    endpoint.result_text = "https://example.org/source " + "web evidence " * 10000
    monkeypatch.setattr("raglite._rag.get_context_size", lambda _: 4096)
    monkeypatch.setattr(
        "raglite._rag._run_tools",
        lambda *_, **__: [
            {"role": "tool", "tool_call_id": "local-call", "content": "Local document result"},
        ],
    )
    messages = [{"role": "user", "content": "Find local and web release facts"}]
    answer = "".join(rag(messages, config=config, web_search=ParallelSearchSession()))
    assert "Local document result" in answer
    web_result = next(
        message["content"] for message in messages if message.get("tool_call_id") == "web-call"
    )
    assert "[Web result truncated to fit context.]" in web_result
    assert len(web_result) < len(endpoint.result_text)
    assert "https://example.org/source" in web_result
    assert any(
        message.get("tool_call_id") == "local-call" for message in model.requests[-1]["messages"]
    )
