"""Turning a Decillion project's agents into a CrewAI crew.

A Decillion agent is a market listing: a name, a descriptor carrying its system
instruction, and the model it was published against. A CrewAI agent is a role,
a goal, a backstory and an LLM. The mapping is direct, and deliberately has no
second source of truth — everything comes from the roster the crew creature
sends with each prompt, so an agent added to the project a second ago is on the
team for the very next turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crewai import Agent

logger = logging.getLogger(__name__)

#: Decillion provider id → the prefix CrewAI's LLM layer expects. An id with no
#: entry is passed through unchanged, which is what a model string like
#: "openai/gpt-4o" already is. Keep this in step with the platform's provider
#: list (`new-decillion/src/api/admin.ts`) and the advisor's endpoint table.
_PROVIDER_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "gemini": "gemini",
    "groq": "groq",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "xai": "xai",
    "openrouter": "openrouter",
    "agentrouter": "openai",  # an OpenAI-compatible gateway
}

#: The placeholder credential the platform's proxy is addressed with. LiteLLM
#: refuses to call an OpenAI-compatible endpoint with no key at all, and there
#: is no real key in this sandbox to give it — that is the whole point.
PROXY_API_KEY = "decillion-proxy"


def model_ref(llm: dict[str, Any] | None) -> str | None:
    """The model string CrewAI should run an agent on.

    Returns `None` when the listing named no model, which leaves CrewAI on its
    own default rather than inventing one — an agent silently switched to a
    different vendor's model is worse than an agent that reports it has none.
    """
    if not llm:
        return None
    model = str(llm.get("model") or "").strip()
    if not model:
        return None
    if "/" in model:
        return model
    provider = str(llm.get("provider") or "").strip().lower()
    prefix = _PROVIDER_PREFIX.get(provider)
    return f"{prefix}/{model}" if prefix else model


def llm_kwargs(llm: dict[str, Any] | None, proxy_base_url: str) -> dict[str, Any]:
    """Where one agent's model calls go.

    Every model an agent runs on is addressed through the platform's proxy, and
    NO credential is passed: this sandbox holds none. The proxy is a loopback
    endpoint served by the bridge, which forwards each call over its socket to
    the `llm` creature — the only place a provider key is ever read, and the
    place that records what the provider said the call cost.

    Handing the key to the sandbox instead (which is what used to happen) put
    the platform's account one container escape away from anyone who could get
    an agent to run a shell, and left the token counts that a run is billed on
    being reported by the very process being billed.

    Without a proxy — a runtime older than one, or one that failed to start —
    no kwargs are returned and LiteLLM falls back to the process environment,
    which in this sandbox has no key either. An agent that cannot reach a model
    says so; it does not quietly reach one unmetered.
    """
    if not llm or not proxy_base_url:
        return {}
    return {"api_key": PROXY_API_KEY, "base_url": proxy_base_url}


#: Told to every agent that has tools. Deliberately about consequences rather
#: than instructions — a model follows "this is what is real" better than "you
#: must", and the point is genuinely a fact about the platform.
_TOOL_PREAMBLE = (
    "You are working on this project's own machine, and you have tools that act on it. "
    "The project folder is where the project's files live: what you write there is what "
    "your teammates see, what the people on this project open in the Files panel, and "
    "what survives after this conversation. Text in an answer is not a delivered file. "
    "When the work you were asked for IS a file — a document, a script, a report — write "
    "it with your file tools and then say where you put it. When you need a fact about "
    "the project, read it with your tools rather than assuming it."
)


#: Told to the lead when it runs a crew as its MANAGER.
#:
#: CrewAI forbids a manager_agent from holding tools — it delegates, it does not
#: execute — so the lead in a led turn genuinely has none, and telling it
#: otherwise is how it ends a turn by asking a question nothing is listening
#: for. That is not hypothetical: a lead asked "Formal or Playful?" as its final
#: answer, because it had been told it could ask and had no tool to ask with.
_MANAGER_PREAMBLE = (
    "You are leading this project's crew. You do not use tools yourself — your "
    "teammates do the work, and they have every tool this project has: its files, its "
    "machine, its connected accounts, and the ability to ask the people on this "
    "project a question and wait for their answer. So delegate the work, and when a "
    "decision is the project's to make, delegate the ASKING too: tell a teammate to "
    "use ask_the_project and report back what they were told. Never end your turn by "
    "asking a question yourself — your turn is the answer, and nobody is waiting to "
    "reply to it."
)


def as_manager(agent: "Agent") -> "Agent":
    """The same agent, prepared to run a crew rather than to work in one.

    Two things change together, and they have to: a manager holds no tools (the
    framework refuses one that does), and it must not be told that it has any.
    The tools preamble is swapped for the manager's rather than removed, because
    an agent that is simply told nothing about tools reaches for them anyway.
    """
    story = str(getattr(agent, "backstory", "") or "")
    if _TOOL_PREAMBLE in story:
        story = story.replace(_TOOL_PREAMBLE, _MANAGER_PREAMBLE)
    return agent.model_copy(update={"tools": [], "backstory": story})


def build_agent(
    spec: dict[str, Any],
    universal_prompt: str,
    proxy_base_url: str,
    tools: list[Any] | None = None,
) -> "Agent":
    """One Decillion agent, as a CrewAI agent.

    The platform's universal instruction goes **before** the agent's own
    persona, exactly as the platform specifies: an admin's edit reaches every
    agent's next turn without any agent being edited or redeployed, and the
    agent's own voice is what the model reads last.
    """
    # CrewAI is imported lazily: it pulls in the whole agent stack, and the
    # pure mapping above (which is what most callers need) should not pay for
    # it — nor should a test of that mapping require the framework.
    from crewai import LLM, Agent

    backstory = str(spec.get("backstory") or spec.get("instruction") or "").strip()
    if tools:
        # What having tools MEANS here, said once.
        #
        # A model asked to "create a file" will happily answer with the file's
        # contents and consider the job done — which is precisely the failure
        # this platform kept hitting: an agent reported saving a document that
        # was never written, and the project's Files panel stayed empty. Having
        # the tool is not enough; the agent has to know that the tool is what
        # counts as delivery.
        #
        # It sits between the platform's instruction and the agent's own voice,
        # because it describes the workplace rather than the work: it is true of
        # every agent on every project, and it must not be the last thing the
        # model reads.
        backstory = f"{_TOOL_PREAMBLE}\n\n{backstory}".strip()
    if universal_prompt:
        backstory = f"{universal_prompt.strip()}\n\n{backstory}".strip()

    llm_spec = spec.get("llm") or {}
    model = model_ref(llm_spec)
    llm = None
    if model:
        try:
            llm = LLM(model=model, **llm_kwargs(llm_spec, proxy_base_url))
        except Exception:  # noqa: BLE001 - a bad model must not lose the agent
            logger.exception("could not build LLM %s; falling back to the default", model)

    return Agent(
        role=str(spec.get("role") or spec.get("name") or "specialist"),
        goal=str(spec.get("goal") or spec.get("name") or "help the project"),
        backstory=backstory or "You are a helpful agent on a Decillion project.",
        llm=llm,
        tools=tools or [],
        verbose=False,
        # Teammates are reached by @mentioning them in an answer, which the
        # platform turns into a fresh turn for that agent. Letting CrewAI
        # delegate as well would run the same hand-off twice, by two different
        # mechanisms, with two different billing paths.
        allow_delegation=False,
    )


def build_roster(
    specs: list[dict[str, Any]],
    universal_prompt: str,
    proxy_base_url: str,
    tools: list[Any] | None = None,
) -> dict[str, "Agent"]:
    """Every agent on the project, keyed by its Decillion program id.

    The program id is the identity everything else uses — it is what a mention
    resolves to and what a message is attributed to — so it is the key here
    too.

    Every agent gets the SAME tools, and deliberately: they are the project's
    tools, not the agent's. The project's files, its machine and its connected
    accounts belong to the space, so which of them an agent may touch is a
    property of the project rather than of the listing somebody published to the
    market. What differs between two agents is the instruction that tells them
    what to do with the tools, which is exactly where the difference belongs.
    """
    roster: dict[str, "Agent"] = {}
    for spec in specs:
        program_id = str(spec.get("programId") or "").strip()
        if not program_id:
            continue
        try:
            roster[program_id] = build_agent(spec, universal_prompt, proxy_base_url, tools)
        except Exception:  # noqa: BLE001 - one bad listing must not empty the team
            logger.exception("skipping agent %s: could not be built", program_id)
    return roster
