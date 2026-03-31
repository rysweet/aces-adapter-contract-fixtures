from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import ChatMessageUser

from aces_adapter_contract_fixtures.briefing import AGENT_CAPABILITIES, create_agent


def test_briefing_declares_no_tool_support() -> None:
    assert AGENT_CAPABILITIES.supports_tools is False


@pytest.mark.asyncio
async def test_briefing_solver_calls_model_without_tools() -> None:
    solver = create_agent(role="briefing")(
        instruction_prompt="Investigate the incident.",
        assistant_prompt="Be concise.",
        max_steps=5,
    )

    model = MagicMock()
    output = MagicMock()
    output.message = MagicMock()
    model.generate = AsyncMock(return_value=output)

    state: Any = SimpleNamespace(
        messages=[ChatMessageUser(content="Start with the key facts.")],
        output=None,
    )

    with patch("aces_adapter_contract_fixtures.briefing.get_model", return_value=model):
        result = await cast(Any, solver)(state, object())

    assert result is state
    call_kwargs = model.generate.await_args.kwargs
    assert call_kwargs["tools"] == []
    input_messages = call_kwargs["input"]
    assert input_messages[0].content.startswith("Role: briefing")
    assert "Investigate the incident." in input_messages[0].content
    assert "Be concise." in input_messages[0].content
    assert state.messages[-1] is output.message
