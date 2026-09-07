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

#: Providers reached through an OpenAI-compatible gateway rather than their own
#: endpoint. These need a base URL as well as a key.
_GATEWAY_BASE_URL = {
    "agentrouter": "https://agentrouter.org/v1",
}


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


def llm_kwargs(llm: dict[str, Any] | None, keys: dict[str, str]) -> dict[str, Any]:
    """Credentials and endpoint for one agent's model.

    `keys` are the platform keys the crew creature sent, narrowed to the
    providers this project's agents actually use. A provider with no key gets
    no kwargs and CrewAI falls back to the process environment.
    """
    if not llm:
        return {}
    provider = str(llm.get("provider") or "").strip().lower()
    out: dict[str, Any] = {}
    api_key = keys.get(provider)
    if api_key:
        out["api_key"] = api_key
    base_url = _GATEWAY_BASE_URL.get(provider)
    if base_url:
        out["base_url"] = base_url
    return out


def build_agent(
    spec: dict[str, Any],
    universal_prompt: str,
    llm_keys: dict[str, str],
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
    if universal_prompt:
        backstory = f"{universal_prompt.strip()}\n\n{backstory}".strip()

    llm_spec = spec.get("llm") or {}
    model = model_ref(llm_spec)
    llm = None
    if model:
        try:
            llm = LLM(model=model, **llm_kwargs(llm_spec, llm_keys))
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
    llm_keys: dict[str, str],
) -> dict[str, "Agent"]:
    """Every agent on the project, keyed by its Decillion program id.

    The program id is the identity everything else uses — it is what a mention
    resolves to and what a message is attributed to — so it is the key here
    too.
    """
    roster: dict[str, "Agent"] = {}
    for spec in specs:
        program_id = str(spec.get("programId") or "").strip()
        if not program_id:
            continue
        try:
            roster[program_id] = build_agent(spec, universal_prompt, llm_keys)
        except Exception:  # noqa: BLE001 - one bad listing must not empty the team
            logger.exception("skipping agent %s: could not be built", program_id)
    return roster
