"""Prompt-only external adapter fixture."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from inspect_ai.model import ChatMessageSystem, get_model
from saber.agents.models import AgentCapabilities

if TYPE_CHECKING:
    from inspect_ai.solver import Generate, Solver, TaskState
    from inspect_ai.tool import Tool

AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)


def _build_system_prompt(
    instruction_prompt: str,
    assistant_prompt: str,
    role: str,
) -> str:
    parts = [f"Role: {role}"] if role else []
    parts.extend(part for part in (instruction_prompt, assistant_prompt) if part)
    return "\n\n".join(parts)


def create_agent(**kwargs: object) -> Callable[..., Solver]:
    """Create a prompt-only external adapter."""

    role = str(kwargs.get("role", "briefing"))

    def create_with_prompts(
        instruction_prompt: str = "",
        assistant_prompt: str = "",
        tools: Sequence[Tool] | None = None,
        *,
        max_steps: int,
        **extra_kwargs: object,
    ) -> Solver:
        del tools, max_steps, extra_kwargs

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            del generate
            system_prompt = _build_system_prompt(
                instruction_prompt=instruction_prompt,
                assistant_prompt=assistant_prompt,
                role=role,
            )
            messages = list(state.messages)
            if system_prompt:
                messages = [ChatMessageSystem(content=system_prompt)] + messages
            state.output = await get_model().generate(input=messages, tools=[])
            state.messages.append(state.output.message)
            return state

        return solve

    return create_with_prompts
