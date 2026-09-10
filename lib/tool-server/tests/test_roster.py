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


def test_an_agent_with_tools_is_told_what_delivery_means(monkeypatch):
    """The preamble only appears for an agent that can actually act.

    An agent with no tools told to "write it to the project folder" would be
    reading an instruction it cannot follow.
    """
    from decillion_caspar_bridge import roster as r

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

    import sys, types

    fake = types.ModuleType("crewai")
    fake.Agent = FakeAgent
    fake.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "crewai", fake)

    spec = {"programId": "a-1", "name": "Writer", "backstory": "You write well."}
    r.build_agent(spec, "PLATFORM RULES", "", ["a tool"])
    story = captured["backstory"]
    assert story.startswith("PLATFORM RULES")
    assert "Text in an answer is not a delivered file" in story
    # The agent's own voice is still what the model reads last.
    assert story.rstrip().endswith("You write well.")

    captured.clear()
    r.build_agent(spec, "PLATFORM RULES", "", [])
    assert "delivered file" not in captured["backstory"]


def test_a_lead_running_the_crew_is_told_it_delegates_instead():
    """The manager has no tools, so it must not be told it has any.

    A lead carrying the tools preamble with an empty toolset ends its turn by
    asking a question nothing is listening for — which is exactly what it did.
    """
    from decillion_caspar_bridge.roster import _MANAGER_PREAMBLE, _TOOL_PREAMBLE, as_manager

    class FakeAgent:
        def __init__(self, backstory, tools):
            self.backstory = backstory
            self.tools = tools

        def model_copy(self, update=None):
            clone = FakeAgent(self.backstory, self.tools)
            for key, value in (update or {}).items():
                setattr(clone, key, value)
            return clone

    worker = FakeAgent(f"{_TOOL_PREAMBLE}\n\nYou lead the project.", ["a tool"])
    manager = as_manager(worker)
    assert manager.tools == []
    assert _TOOL_PREAMBLE not in manager.backstory
    assert _MANAGER_PREAMBLE in manager.backstory
    # The agent's own persona survives the swap.
    assert manager.backstory.rstrip().endswith("You lead the project.")
    # And the original is untouched — it may still run as a worker elsewhere.
    assert worker.tools == ["a tool"]
