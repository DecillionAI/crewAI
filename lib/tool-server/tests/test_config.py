"""Configuration comes from the sandbox, and the token never leaks into logs."""

from decillion_tool_server.config import load_config


def test_topic_defaults_to_the_space(monkeypatch):
    monkeypatch.setenv("DECILLION_SPACE_ID", "42")
    monkeypatch.delenv("DECILLION_BRIDGE_TOPIC", raising=False)
    assert load_config().topic == "space:42"


def test_describe_never_includes_the_token(monkeypatch):
    monkeypatch.setenv("DECILLION_SPACE_ID", "42")
    monkeypatch.setenv("DECILLION_BRIDGE_TOKEN", "super-secret-value")
    monkeypatch.setenv("CASPAR_GATEWAY_URL", "ws://node:8076")
    described = load_config().describe()
    assert "super-secret-value" not in described
    assert "token=set" in described


def test_missing_token_is_not_configured(monkeypatch):
    monkeypatch.setenv("DECILLION_SPACE_ID", "42")
    monkeypatch.setenv("CASPAR_GATEWAY_URL", "ws://node:8076")
    monkeypatch.delenv("DECILLION_BRIDGE_TOKEN", raising=False)
    assert not load_config().configured
