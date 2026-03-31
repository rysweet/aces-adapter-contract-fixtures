from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

from aces_adapter_contract_fixtures.bridge_openai_mcp import (
    _RUNNER_PATH,
    AGENT_CAPABILITIES,
    create_agent,
)


class _Store:
    def __init__(self) -> None:
        self._data: dict[str, int] = {}

    def get(self, key: str, default: int) -> int:
        return self._data.get(key, default)

    def set(self, key: str, value: int) -> None:
        self._data[key] = value


def test_bridge_fixture_declares_tool_support() -> None:
    assert AGENT_CAPABILITIES.supports_tools is True


@pytest.mark.asyncio
async def test_bridge_solver_writes_runner_and_executes_inside_bridge() -> None:
    bridge_calls: list[dict[str, object]] = []
    fake_bridge = SimpleNamespace(
        port=13131,
        state=SimpleNamespace(messages=[{"role": "assistant", "content": "done"}]),
        mcp_server_configs=[
            MCPServerConfigHTTP(
                name="saber_tools",
                type="http",
                url="http://localhost:13131/mcp/saber_tools",
                tools="all",
            )
        ],
    )

    @asynccontextmanager
    async def fake_bridge_cm(*args: object, **kwargs: object) -> AsyncIterator[SimpleNamespace]:
        bridge_calls.append({"args": args, "kwargs": kwargs})
        yield fake_bridge

    fake_sbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
    )
    fake_store = _Store()

    with (
        patch("inspect_ai.agent.agent", side_effect=lambda fn: fn),
        patch("inspect_ai.agent.as_solver", side_effect=lambda agent, limits=None: {"agent": agent, "limits": limits}),
        patch("inspect_ai.agent.sandbox_agent_bridge", side_effect=fake_bridge_cm),
        patch("inspect_ai.util.sandbox", return_value=fake_sbox),
        patch("inspect_ai.util.store", return_value=fake_store),
        patch("aces_adapter_contract_fixtures.bridge_openai_mcp.validate_model_availability", new=AsyncMock()),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.create_tool_call_limit_filter",
            return_value=("bridge-filter", lambda: None),
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.create_tracking_filter",
            return_value=("tracking-filter", lambda: {"events": 1}),
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.compose_filters",
            return_value="composed-filter",
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.build_user_prompt",
            return_value=("Use the bridge.", False),
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.build_bridged_tools",
            return_value=[{"name": "saber_tools"}],
        ),
        patch("aces_adapter_contract_fixtures.bridge_openai_mcp.record_bridge_summary"),
        patch("aces_adapter_contract_fixtures.bridge_openai_mcp.tool_call_limit", return_value="limit:7"),
    ):
        solver_info = cast(
            Any,
            create_agent(sandbox_name="fixture-sandbox")(
                instruction_prompt="Inspect the environment.",
                assistant_prompt="Use tools when needed.",
                tools=[MagicMock()],
                max_steps=7,
            ),
        )
        result = await solver_info["agent"](SimpleNamespace(messages=[], output=None))

    assert result is fake_bridge.state
    assert solver_info["limits"] == ["limit:7"]
    bridge_kwargs = cast(dict[str, Any], bridge_calls[0]["kwargs"])
    assert bridge_kwargs["sandbox"] == "fixture-sandbox"
    assert bridge_kwargs["bridged_tools"] == [{"name": "saber_tools"}]
    fake_sbox.write_file.assert_awaited_once()
    write_args = fake_sbox.write_file.await_args.args
    assert write_args[0] == _RUNNER_PATH
    assert "BRIDGE_MCP_CONFIG" in write_args[1]

    exec_env = fake_sbox.exec.await_args.kwargs["env"]
    assert exec_env["OPENAI_BASE_URL"] == "http://localhost:13131/v1"
    assert exec_env["BRIDGE_MAX_TOOL_CALLS"] == "7"
    mcp_config = json.loads(exec_env["BRIDGE_MCP_CONFIG"])
    assert mcp_config[0]["name"] == "saber_tools"


@pytest.mark.asyncio
async def test_bridge_solver_propagates_runner_failures() -> None:
    fake_bridge = SimpleNamespace(
        port=13131,
        state=SimpleNamespace(messages=[{"role": "assistant", "content": "partial"}]),
        mcp_server_configs=[],
    )

    @asynccontextmanager
    async def fake_bridge_cm(*args: object, **kwargs: object) -> AsyncIterator[SimpleNamespace]:
        del args, kwargs
        yield fake_bridge

    fake_sbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="runner exploded")),
    )
    fake_store = _Store()

    with (
        patch("inspect_ai.agent.agent", side_effect=lambda fn: fn),
        patch("inspect_ai.agent.as_solver", side_effect=lambda agent, limits=None: {"agent": agent, "limits": limits}),
        patch("inspect_ai.agent.sandbox_agent_bridge", side_effect=fake_bridge_cm),
        patch("inspect_ai.util.sandbox", return_value=fake_sbox),
        patch("inspect_ai.util.store", return_value=fake_store),
        patch("aces_adapter_contract_fixtures.bridge_openai_mcp.validate_model_availability", new=AsyncMock()),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.create_tool_call_limit_filter",
            return_value=("bridge-filter", lambda: None),
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.create_tracking_filter",
            return_value=("tracking-filter", lambda: {"events": 1}),
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.compose_filters",
            return_value="composed-filter",
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.build_user_prompt",
            return_value=("Use the bridge.", False),
        ),
        patch(
            "aces_adapter_contract_fixtures.bridge_openai_mcp.build_bridged_tools",
            return_value=[],
        ),
        patch("aces_adapter_contract_fixtures.bridge_openai_mcp.record_bridge_summary"),
        patch("aces_adapter_contract_fixtures.bridge_openai_mcp.tool_call_limit", return_value="limit:7"),
    ):
        solver_info = cast(
            Any,
            create_agent(sandbox_name="fixture-sandbox")(
                instruction_prompt="Inspect the environment.",
                assistant_prompt="Use tools when needed.",
                tools=[],
                max_steps=7,
            ),
        )

        with pytest.raises(RuntimeError, match="runner exploded"):
            await solver_info["agent"](SimpleNamespace(messages=[], output=None))
