"""A Decillion listing becomes a CrewAI agent with no second source of truth."""

from decillion_caspar_bridge.roster import llm_kwargs, model_ref


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


def test_gateway_providers_get_a_base_url_with_their_key():
    kwargs = llm_kwargs({"provider": "agentrouter"}, {"agentrouter": "k"})
    assert kwargs["api_key"] == "k"
    assert kwargs["base_url"] == "https://agentrouter.org/v1"


def test_a_provider_without_a_key_contributes_nothing():
    assert llm_kwargs({"provider": "openai"}, {}) == {}
