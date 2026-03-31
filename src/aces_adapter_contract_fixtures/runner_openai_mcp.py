#!/usr/bin/env python3
"""Minimal in-sandbox runner for the bridge fixture adapter.

This script uses only the Python standard library so it can run inside a task
sandbox without assuming extra Python packages are installed.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

JSON_CONTENT_TYPE = "application/json"
JSON_AND_SSE_ACCEPT = "application/json, text/event-stream"
LATEST_PROTOCOL_VERSION = "2025-11-25"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, str], str, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            content_type = response_headers.get("content-type", "")
            return response.status, response_headers, body, content_type
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        response_headers = {key.lower(): value for key, value in exc.headers.items()}
        content_type = response_headers.get("content-type", "")
        return exc.code, response_headers, body, content_type


class MCPClient:
    def __init__(self, *, name: str, url: str) -> None:
        self.name = name
        self.url = url
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._request_id = 0

    def _headers(self) -> dict[str, str]:
        headers = {
            "accept": JSON_AND_SSE_ACCEPT,
            "content-type": JSON_CONTENT_TYPE,
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        if self.protocol_version:
            headers["mcp-protocol-version"] = self.protocol_version
        return headers

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        expect_response: bool = True,
    ) -> dict[str, Any]:
        self._request_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        status, headers, body, content_type = _post_json(self.url, payload, self._headers())
        if status >= 400:
            raise RuntimeError(f"MCP request {method!r} failed with HTTP {status}: {body[:500]}")
        if not self.session_id:
            self.session_id = headers.get("mcp-session-id")
        if not expect_response or status in (202, 204) or not body.strip():
            return {}
        if not content_type.lower().startswith(JSON_CONTENT_TYPE):
            raise RuntimeError(
                f"MCP request {method!r} returned unsupported content-type {content_type!r}; "
                "this runner only supports JSON responses."
            )
        response = json.loads(body)
        if "error" in response:
            raise RuntimeError(f"MCP request {method!r} returned error: {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"MCP request {method!r} returned non-object result: {response}")
        return result

    def initialize(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "bridge-fixture-runner", "version": "0.1.0"},
            },
        )
        self.protocol_version = str(result.get("protocolVersion", LATEST_PROTOCOL_VERSION))
        self._notify("notifications/initialized")

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        _post_json(self.url, payload, self._headers())

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        return list(result.get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": arguments})


def _tool_schema(tool: dict[str, Any], alias: str) -> dict[str, Any]:
    parameters = tool.get("inputSchema") or {"type": "object", "properties": {}}
    if "type" not in parameters:
        parameters = {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "additionalProperties": True,
        }
    return {
        "type": "function",
        "function": {
            "name": alias,
            "description": tool.get("description") or alias,
            "parameters": parameters,
        },
    }


def _flatten_tool_result(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in result.get("content", []):
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "resource":
            parts.append(json.dumps(block.get("resource", {}), ensure_ascii=True))
        else:
            parts.append(json.dumps(block, ensure_ascii=True))
    if result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=True))
    if not parts:
        parts.append(json.dumps(result, ensure_ascii=True))
    return "\n".join(part for part in parts if part)


def _load_tool_catalog(configs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, tuple[MCPClient, str]]]:
    openai_tools: list[dict[str, Any]] = []
    tool_map: dict[str, tuple[MCPClient, str]] = {}

    for config in configs:
        client = MCPClient(name=str(config["name"]), url=str(config["url"]))
        client.initialize()
        for tool in client.list_tools():
            tool_name = str(tool["name"])
            alias = f"{client.name}__{tool_name}"
            openai_tools.append(_tool_schema(tool, alias))
            tool_map[alias] = (client, tool_name)

    return openai_tools, tool_map


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    status, _headers, body, content_type = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        payload,
        {
            "accept": JSON_CONTENT_TYPE,
            "content-type": JSON_CONTENT_TYPE,
            "authorization": f"Bearer {api_key}",
        },
    )
    if status >= 400:
        raise RuntimeError(f"OpenAI bridge request failed with HTTP {status}: {body[:500]}")
    if not content_type.lower().startswith(JSON_CONTENT_TYPE):
        raise RuntimeError(f"Unexpected OpenAI bridge content-type: {content_type!r}")
    response = json.loads(body)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI bridge response had no choices: {response}")
    return dict(choices[0].get("message") or {})


def main() -> int:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:13131/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder-for-bridge")
    model = os.environ.get("BRIDGE_MODEL", "inspect")
    prompt = os.environ.get("BRIDGE_PROMPT", "")
    max_tool_calls = int(os.environ.get("BRIDGE_MAX_TOOL_CALLS", "20"))
    raw_mcp_config = os.environ.get("BRIDGE_MCP_CONFIG", "[]")

    if not prompt:
        print("ERROR: BRIDGE_PROMPT is required", file=sys.stderr)
        return 1

    try:
        mcp_configs = json.loads(raw_mcp_config)
        if not isinstance(mcp_configs, list):
            raise ValueError("BRIDGE_MCP_CONFIG must decode to a list")
        tools, tool_map = _load_tool_catalog(mcp_configs)

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        executed_tool_calls = 0

        while True:
            assistant_message = _chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                tools=tools,
            )
            tool_calls = assistant_message.get("tool_calls") or []
            normalized_assistant = {
                "role": "assistant",
                "content": assistant_message.get("content") or "",
            }
            if tool_calls:
                normalized_assistant["tool_calls"] = tool_calls
            messages.append(normalized_assistant)

            if not tool_calls:
                print("BRIDGE_FIXTURE_RUNNER_COMPLETE", file=sys.stderr)
                return 0

            for tool_call in tool_calls:
                if executed_tool_calls >= max_tool_calls:
                    print("BRIDGE_FIXTURE_RUNNER_TOOL_LIMIT_REACHED", file=sys.stderr)
                    return 0

                function = tool_call.get("function") or {}
                alias = str(function.get("name", ""))
                if alias not in tool_map:
                    raise RuntimeError(f"Unknown tool alias from model: {alias!r}")

                raw_arguments = function.get("arguments") or "{}"
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": raw_arguments}
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    arguments = {"raw_arguments": raw_arguments}

                client, tool_name = tool_map[alias]
                result = client.call_tool(tool_name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id", alias)),
                        "content": _flatten_tool_result(result),
                    }
                )
                executed_tool_calls += 1

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
