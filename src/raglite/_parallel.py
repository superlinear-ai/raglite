"""Opt-in, per-conversation web tools from Parallel Search MCP."""

import asyncio
import copy
import json
import math
from datetime import timedelta
from typing import Any
from uuid import uuid4

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class ParallelSearchSession:
    """Enable web search for one RAG conversation.

    Create one instance per conversation and reuse it for follow-up turns with
    either ``rag`` or ``async_rag``. Create a fresh instance for a new chat.
    Queries and requested URLs are sent to Parallel only when the model calls a
    web tool. No API key is needed. ``api_key`` explicitly enables authenticated
    access; environment credentials are never read implicitly.
    """

    def __init__(self, *, timeout: float = 30, api_key: str | None = None) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            message = "Web search timeout must be a positive finite number."
            raise ValueError(message)
        self.timeout = timeout
        self._session_id = str(uuid4())
        self._headers = {"User-Agent": "RAGLite (+https://github.com/superlinear-ai/raglite)"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._tools: list[dict[str, Any]] | None = None

    async def _request(self, name: str | None, arguments: dict[str, Any] | None = None) -> Any:
        async def request() -> Any:
            async with (
                streamablehttp_client(
                    "https://search.parallel.ai/mcp",
                    headers=self._headers,
                    timeout=timedelta(seconds=self.timeout),
                    sse_read_timeout=timedelta(seconds=self.timeout),
                ) as (read_stream, write_stream, _),
                ClientSession(read_stream, write_stream) as client,
            ):
                await client.initialize()
                if name is None:
                    return await client.list_tools()
                return await client.call_tool(name, arguments)

        return await asyncio.wait_for(request(), timeout=self.timeout)

    async def tools(self) -> list[dict[str, Any]]:
        """Discover the two web tools, keeping runtime metadata out of model control."""
        if self._tools is None:
            discovered = await self._request(None)
            tools: list[dict[str, Any]] = []
            for tool in discovered.tools:
                if tool.name not in {"web_search", "web_fetch"}:
                    continue
                schema = copy.deepcopy(tool.inputSchema)
                for parameter in ("session_id", "model_name"):
                    schema.get("properties", {}).pop(parameter, None)
                    if parameter in schema.get("required", []):
                        schema["required"].remove(parameter)
                schema["additionalProperties"] = False
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"parallel_{tool.name}",
                            "description": tool.description or tool.name,
                            "parameters": schema,
                        },
                    }
                )
            if {tool["function"]["name"] for tool in tools} != {
                "parallel_web_search",
                "parallel_web_fetch",
            }:
                message = "Parallel Search MCP did not provide web_search and web_fetch."
                raise ValueError(message)
            self._tools = tools
        return copy.deepcopy(self._tools)

    async def call(self, name: str, arguments: str, *, model_name: str) -> str:
        """Execute a selected web tool with this conversation's trusted metadata."""
        if name not in {"parallel_web_search", "parallel_web_fetch"}:
            message = f"Unknown web tool {name}."
            raise ValueError(message)
        parameters = json.loads(arguments)
        if not isinstance(parameters, dict):
            message = "Web tool arguments must be a JSON object."
            raise TypeError(message)
        parameters["session_id"] = self._session_id
        parameters.pop("model_name", None)
        if len(model_name) <= 100:  # noqa: PLR2004 (MCP metadata limit)
            parameters["model_name"] = model_name
        result = await self._request(name.removeprefix("parallel_"), parameters)
        if result.isError:
            message = "Parallel web tool failed: " + "\n".join(
                item.text for item in result.content if item.type == "text"
            )
            raise ValueError(message)
        return "\n".join(item.text for item in result.content if item.type == "text")
