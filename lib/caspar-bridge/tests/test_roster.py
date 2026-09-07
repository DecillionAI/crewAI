"""A Decillion listing becomes a CrewAI agent with no second source of truth."""

from decillion_caspar_bridge.roster import PROXY_API_KEY, llm_kwargs, model_ref


def test_model_ref_prefixes_a_bare_model_with_its_provider():
    assert model_ref({"provider": "openai", "model": "gpt-4o"}) == "openai/gpt-4o"
    assert model_ref({"provider": "google", "model": "gemini-2.0"}) == "gemini/gemini-2.0"


def test_model_ref_leaves_an_already_qualified_model_alone():
    assert model_ref({"provider": "openai", "model": "anthropic/claude"}) == "anthropic/claude"


def test_model_ref_is_none_when_the_listing_named_no_model():
    # Silently switching an agent to some default vendor is worse than saying
    # it has no model, so this must not invent one.
    assert model_ref({"provider": "openai"}) is None
    assert model_ref(None) is None


def test_unknown_provider_passes_the_model_through_unchanged():
    # An id the table does not know must never fall back to OpenAI — that
    # would post one vendor's key to a different vendor.
    assert model_ref({"provider": "somethingnew", "model": "m1"}) == "m1"


def test_every_model_is_addressed_through_the_platform_proxy_with_no_real_key():
    # A provider key must never be inside the sandbox: an agent that gets a
    # shell here would find the platform's account. Every vendor, gateway or
    # not, is reached the same way — through the bridge.
    kwargs = llm_kwargs({"provider": "agentrouter"}, "http://127.0.0.1:8788/v1")
    assert kwargs["base_url"] == "http://127.0.0.1:8788/v1"
    assert kwargs["api_key"] == PROXY_API_KEY
    assert llm_kwargs({"provider": "openai"}, "http://127.0.0.1:8788/v1") == kwargs


def test_without_a_proxy_nothing_is_configured_rather_than_reaching_a_model_unmetered():
    # No proxy means no metered path. LiteLLM then falls back to the process
    # environment, which in this sandbox holds no key either, so the agent
    # reports that it cannot reach a model instead of quietly reaching one.
    assert llm_kwargs({"provider": "openai"}, "") == {}
