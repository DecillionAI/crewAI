"""A call to a creature waits for the creature's answer, not for delivery."""

import asyncio

import pytest

from decillion_caspar_bridge.client import CasparBridgeClient
from decillion_caspar_bridge.config import BridgeConfig


def _client():
    cfg = BridgeConfig(
        gateway_url="ws://127.0.0.1:1",
        space_id="s1",
        topic="space:s1",
        token="t" * 40,
        crew_home="/opt/crewai",
        log_level="INFO",
        llm_proxy_port=8788,
    )

    async def on_update(key, data):
        return None

    return CasparBridgeClient(cfg, on_update)


def test_a_creature_call_waits_for_the_answer_not_the_acknowledgement():
    client = _client()
    sent = []

    async def fake_signal(action, payload, correlation_id=""):
        sent.append((action, payload))
        # What `/gateway/signal` actually returns: the node took the call. The
        # creature's answer arrives separately, on the project's topic.
        asyncio.get_running_loop().call_soon(
            client.resolve_creature_call,
            payload["correlationId"],
            {"ok": True, "response": {"choices": []}},
        )
        return {"ok": True, "creatureId": "c1", "correlationId": ""}

    client.signal = fake_signal
    out = asyncio.run(client.call_creature("llm/chat", {"model": "m"}))

    # Reading the acknowledgement as the reply is what answered every model
    # call with "the platform returned no completion".
    assert out == {"ok": True, "response": {"choices": []}}
    action, payload = sent[0]
    assert action == "llm/chat"
    assert payload["correlationId"], "the call must carry an id the answer can name"


def test_a_creature_call_that_is_refused_on_delivery_fails_fast():
    client = _client()

    async def fake_signal(action, payload, correlation_id=""):
        return {"ok": False, "error": "token does not grant this topic"}

    client.signal = fake_signal
    with pytest.raises(RuntimeError, match="token does not grant this topic"):
        asyncio.run(client.call_creature("llm/chat", {}))


def test_a_lost_connection_fails_a_waiting_call_instead_of_hanging():
    client = _client()

    async def main():
        async def fake_signal(action, payload, correlation_id=""):
            # The socket dies before the creature answers.
            asyncio.get_running_loop().call_soon(client._fail_pending, "connection closed")
            return {"ok": True}

        client.signal = fake_signal
        with pytest.raises(RuntimeError, match="connection closed"):
            await client.call_creature("llm/chat", {}, timeout=5)

    asyncio.run(main())


def test_an_answer_nobody_is_waiting_for_is_dropped_rather_than_raising():
    client = _client()
    assert client.resolve_creature_call("never-asked", {"ok": True}) is False
