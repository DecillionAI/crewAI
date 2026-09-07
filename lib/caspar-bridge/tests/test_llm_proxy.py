"""The sandbox's model endpoint holds no credential and counts nothing itself."""

import asyncio

from decillion_caspar_bridge.llm_proxy import LlmProxyServer


def _proxy(sent, reply):
    async def send(action, payload):
        sent.append((action, payload))
        return reply

    return LlmProxyServer(send, port=0)


def test_a_call_is_forwarded_with_the_decillion_provider_not_the_litellm_prefix():
    sent = []
    proxy = _proxy(sent, {"ok": True, "response": {"choices": []}})
    # LiteLLM spells an OpenAI-compatible gateway "openai"; the Decillion
    # provider is what says whose key to spend, so it comes from the roster.
    proxy.bind_model("gpt-4o", "agentrouter")
    status, out = asyncio.run(
        proxy._forward({"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    )
    assert status == 200
    assert out == {"choices": []}
    action, payload = sent[0]
    assert action == "llm/chat"
    assert payload["provider"] == "agentrouter"
    assert payload["model"] == "gpt-4o"
    # Nothing resembling a credential leaves this process, because it has none.
    assert "api_key" not in payload and "key" not in payload


def test_a_model_with_no_bound_provider_is_refused_rather_than_guessed():
    sent = []
    proxy = _proxy(sent, {"ok": True})
    status, out = asyncio.run(proxy._forward({"model": "mystery", "messages": []}))
    assert status == 400
    assert "mystery" in out["error"]["message"]
    assert sent == []


def test_the_providers_own_status_and_message_are_relayed():
    # A rate limit and a bad model id need different responses from whoever is
    # looking at it, so a proxy must not flatten both into "the model failed".
    proxy = _proxy([], {"ok": False, "status": 429, "error": "rate limited"})
    proxy.bind_model("m", "openai")
    status, out = asyncio.run(proxy._forward({"model": "m", "messages": []}))
    assert status == 429
    assert out["error"]["message"] == "rate limited"


def test_streaming_is_asked_for_on_the_creature_side_only():
    # A creature answers a signal once, so there is no stream to hand back —
    # but asking for one still matters: several providers report token usage
    # only on a stream's final event.
    sent = []
    proxy = _proxy(sent, {"ok": True, "response": {}})
    proxy.bind_model("m", "openai")
    asyncio.run(proxy._forward({"model": "m", "messages": [], "stream": True}))
    assert sent[0][1]["stream"] is True
