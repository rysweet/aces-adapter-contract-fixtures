"""Bridge-managed external adapter fixture using a custom sandbox runner."""

from __future__ import annotations

import importlib.resources as resources
import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from saber.ext import (
    AgentCapabilities,
    build_bridged_tools,
    build_system_prompt,
    build_user_prompt,
    compose_filters,
    create_tool_call_limit_filter,
    create_tracking_filter,
    parse_bridge_stderr,
    record_bridge_summary,
    resolve_model_aliases,
    tool_call_limit,
    validate_model_availability,
)

if TYPE_CHECKING:
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool
    from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

logger = logging.getLogger(__name__)

AGENT_CAPABILITIES = AgentCapabilities(supports_tools=True)

_STORE_PORT_KEY = "fixture_bridge_openai_mcp_port"
_DEFAULT_PORT_BASE = 3400
_RUNNER_PATH = "/tmp/_bridge_openai_mcp_runner.py"


class BridgeOpenAIMCPConfig(BaseModel):
    """Minimal bridge configuration for the external fixture adapter."""

    model_config = ConfigDict(frozen=True)

    sandbox_name: str = "default"
    port_base: int = _DEFAULT_PORT_BASE
    model: str = "inspect"

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, object]) -> BridgeOpenAIMCPConfig:
        return cls.model_validate({k: v for k, v in kwargs.items() if k in cls.model_fields})


def _runner_source() -> str:
    return (
        resources.files("aces_adapter_contract_fixtures")
        .joinpath("runner_openai_mcp.py")
        .read_text(encoding="utf-8")
    )


def _build_runner_env(
    *,
    bridge_port: int,
    model: str,
    prompt: str,
    mcp_configs: Sequence[MCPServerConfigHTTP],
    max_tool_calls: int,
) -> dict[str, str]:
    return {
        "OPENAI_BASE_URL": f"http://localhost:{bridge_port}/v1",
        "OPENAI_API_KEY": "sk-placeholder-for-bridge",
        "BRIDGE_MODEL": model,
        "BRIDGE_PROMPT": prompt,
        "BRIDGE_MAX_TOOL_CALLS": str(max_tool_calls),
        "BRIDGE_MCP_CONFIG": json.dumps([cfg.model_dump(mode="json") for cfg in mcp_configs]),
    }


def create_agent(**kwargs: object) -> Callable[..., Solver]:
    """Create a bridge-managed external fixture adapter."""

    outer_kwargs = dict(kwargs)
    config = BridgeOpenAIMCPConfig.from_kwargs(outer_kwargs)

    def create_with_prompts(
        instruction_prompt: str = "",
        assistant_prompt: str = "",
        tools: Sequence[Tool] | None = None,
        *,
        max_steps: int,
        **extra_kwargs: object,
    ) -> Solver:
        del extra_kwargs
        from inspect_ai.agent import Agent, AgentState, agent, as_solver, sandbox_agent_bridge
        from inspect_ai.util import sandbox as sandbox_env
        from inspect_ai.util import store

        system_prompt = build_system_prompt(
            instruction_prompt=instruction_prompt,
            assistant_prompt=assistant_prompt,
        )
        bridged_tools = build_bridged_tools(tools)

        @agent
        def _bridge_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                await validate_model_availability()

                port = store().get(_STORE_PORT_KEY, config.port_base) + 1
                store().set(_STORE_PORT_KEY, port)

                bridge_filter, check_tool_limit = create_tool_call_limit_filter()
                tracking_filter, get_tracking_summary = create_tracking_filter()
                composed_filter = compose_filters(tracking_filter, bridge_filter)

                raw_aliases = outer_kwargs.get("model_aliases")
                model_aliases = resolve_model_aliases(raw_aliases if isinstance(raw_aliases, dict) else None)

                async with sandbox_agent_bridge(
                    state,
                    model=config.model,
                    port=port,
                    sandbox=config.sandbox_name,
                    bridged_tools=bridged_tools,
                    filter=composed_filter,
                    model_aliases=model_aliases,
                ) as bridge:
                    sbox = sandbox_env(config.sandbox_name)
                    await sbox.write_file(_RUNNER_PATH, _runner_source())

                    user_prompt, _has_assistant_response = build_user_prompt(state.messages)
                    if not user_prompt:
                        user_prompt = "Begin the task."
                    if system_prompt:
                        user_prompt = f"{system_prompt}\n\n{user_prompt}"

                    runner_env = _build_runner_env(
                        bridge_port=bridge.port,
                        model=config.model,
                        prompt=user_prompt,
                        mcp_configs=bridge.mcp_server_configs,
                        max_tool_calls=max_steps,
                    )

                    result = await sbox.exec(
                        [
                            "python3",
                            _RUNNER_PATH,
                        ],
                        env=runner_env,
                    )

                    if result.returncode != 0:
                        stderr = result.stderr or ""
                        diagnostics = parse_bridge_stderr(stderr)
                        if diagnostics:
                            logger.error("Bridge fixture runner diagnostics:\n%s", diagnostics)
                        stdout = result.stdout or ""
                        detail = stderr[:500] if stderr else stdout[:500] if stdout else "(no output)"
                        raise RuntimeError(f"Bridge fixture runner exited with code {result.returncode}: {detail}")

                    if result.stderr:
                        warnings = parse_bridge_stderr(result.stderr)
                        if warnings:
                            logger.warning("Bridge fixture runner warnings:\n%s", warnings)

                record_bridge_summary(get_tracking_summary, logger)
                check_tool_limit()
                return bridge.state

            return execute

        return as_solver(_bridge_agent(), limits=[tool_call_limit(max_steps)])

    return create_with_prompts
