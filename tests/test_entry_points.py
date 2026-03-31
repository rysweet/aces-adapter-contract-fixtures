from importlib.metadata import entry_points


def test_saber_agent_entry_points_are_registered() -> None:
    names = {entry_point.name for entry_point in entry_points(group="saber.agents")}
    assert "fixture_briefing" in names
    assert "fixture_bridge_openai_mcp" in names
