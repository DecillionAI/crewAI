"""Streaming a crew's work back into the project's chat.

CrewAI publishes everything a run does on its event bus. Subscribing to it is
what makes an agent's work visible in Decillion while it happens rather than
only at the end — and it is the whole of the integration: nothing inside CrewAI
is patched, and a CrewAI upgrade that adds events adds them here for free.

The mapping to Decillion's signal vocabulary is one-to-one:

    tool used            → kind=toolcall
    task started/ended   → kind=step
    agent finished       → kind=step   (the run's answer is posted separately)

`kind=answer` is deliberately NOT emitted here. The runtime posts exactly one
answer per turn, after the crew returns, because the platform's rule is one
writer per record — two paths writing the same row is how a transcript ends up
with duplicates nobody can reconcile.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: How a step reaches the node. Set by the runtime; takes (kind, payload).
Emitter = Callable[[str, dict[str, Any]], None]


class CrewEventForwarder:
    """Forwards one run's CrewAI events to the project's chat.

    Registered per run rather than per process so a step is always attributed
    to the run and agent that produced it — several turns can be in flight in
    one sandbox, and a global listener could not tell them apart.
    """

    def __init__(
        self,
        emit: Emitter,
        run_id: str,
        agent_program_id: str,
        agent_name: str = "",
        *,
        sources: Iterable[Any] = (),
        actors: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        self._emit = emit
        self._run_id = run_id
        self._agent_program_id = agent_program_id
        self._agent_name = agent_name
        self._source_ids = {id(source) for source in sources if source is not None}
        self._actors = dict(actors or {})
        self._registered: list[tuple[type[Any], Any]] = []
        self._bus: Any = None

    def register(self) -> None:
        """Subscribe to the events this bridge forwards.

        Import failures are survivable on purpose: a CrewAI release that
        renames an event should cost the live trail, not the run.
        """
        try:
            from crewai.events import crewai_event_bus
            from crewai.events.types.task_events import (
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )
            from crewai.events.types.tool_usage_events import (
                ToolUsageErrorEvent,
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
            )
        except ImportError:  # pragma: no cover - depends on the CrewAI version
            logger.warning("crewai event bus unavailable; work will not stream live")
            return

        bus = crewai_event_bus
        self._bus = bus

        @bus.on(TaskStartedEvent)
        def _task_started(_source: Any, event: Any) -> None:
            if not self._owns(_source, event):
                return
            self._step("started", getattr(event, "task", None))

        @bus.on(TaskCompletedEvent)
        def _task_completed(_source: Any, event: Any) -> None:
            if not self._owns(_source, event):
                return
            self._step("completed", getattr(event, "task", None), getattr(event, "output", None))

        @bus.on(TaskFailedEvent)
        def _task_failed(_source: Any, event: Any) -> None:
            if not self._owns(_source, event):
                return
            self._step("failed", getattr(event, "task", None), getattr(event, "error", None))

        @bus.on(ToolUsageStartedEvent)
        def _tool_started(_source: Any, event: Any) -> None:
            if not self._owns(_source, event):
                return
            self._tool("started", event)

        @bus.on(ToolUsageFinishedEvent)
        def _tool_finished(_source: Any, event: Any) -> None:
            if not self._owns(_source, event):
                return
            self._tool("finished", event)

        @bus.on(ToolUsageErrorEvent)
        def _tool_error(_source: Any, event: Any) -> None:
            if not self._owns(_source, event):
                return
            self._tool("error", event)

        self._registered = [
            (TaskStartedEvent, _task_started),
            (TaskCompletedEvent, _task_completed),
            (TaskFailedEvent, _task_failed),
            (ToolUsageStartedEvent, _tool_started),
            (ToolUsageFinishedEvent, _tool_finished),
            (ToolUsageErrorEvent, _tool_error),
        ]

    def unregister(self) -> None:
        """Remove every handler installed by this run.

        The CrewAI event bus is process-global. Leaving a run's handlers behind
        makes every later task appear under every earlier run id and grows work
        fan-out without bound.
        """
        bus = self._bus
        if bus is not None:
            for event_type, handler in self._registered:
                bus.off(event_type, handler)
        self._registered = []
        self._bus = None

    def _owns(self, source: Any, event: Any) -> bool:
        """Whether an event belongs to this run's Crew/Task/Agent objects."""
        if not self._source_ids:
            return True
        task = getattr(event, "task", None) or getattr(event, "from_task", None)
        agent = (
            getattr(event, "agent", None)
            or getattr(event, "from_agent", None)
            or getattr(task, "agent", None)
        )
        return any(
            candidate is not None and id(candidate) in self._source_ids
            for candidate in (source, task, agent)
        )

    def _actor(self, source: Any) -> tuple[str, str]:
        if source is None:
            return "", ""
        agent = getattr(source, "agent", None) or source
        if identity := self._actors.get(id(agent)):
            return identity
        return "", _actor_of(agent)

    # ── emitters ─────────────────────────────────────────────────────────

    def _step(self, state: str, task: Any, detail: Any = None) -> None:
        description = _text(getattr(task, "description", "")) if task else ""
        payload = {
            "runId": self._run_id,
            "agentProgramId": self._agent_program_id,
            "agentName": self._agent_name,
            "status": state,
            "text": description,
            "data": {"detail": _text(detail)} if detail is not None else None,
        }
        # Who actually did this. On a led turn the run belongs to the lead but
        # the work is done by whichever teammate it delegated to, so without
        # this the whole crew's trail reads as the lead doing everything.
        actor_program_id, actor_name = self._actor(task)
        if actor_program_id:
            payload["actorProgramId"] = actor_program_id
        if actor_name:
            payload["actorName"] = actor_name
        self._emit("step", payload)

    def _tool(self, state: str, event: Any) -> None:
        payload = {
            "runId": self._run_id,
            "agentProgramId": self._agent_program_id,
            "agentName": self._agent_name,
            "status": state,
            "toolName": getattr(event, "tool_name", "") or "",
            "toolArgs": _jsonable(getattr(event, "tool_args", None)),
            "toolResult": _text(getattr(event, "output", None)),
            "text": getattr(event, "tool_name", "") or "",
        }
        actor_program_id, actor_name = self._actor(getattr(event, "agent", None) or event)
        if actor_program_id:
            payload["actorProgramId"] = actor_program_id
        if actor_name:
            payload["actorName"] = actor_name
        self._emit("toolcall", payload)


def _actor_of(source: Any) -> str:
    """The role of the agent behind an event, when CrewAI names one.

    A task carries the agent assigned to it; a tool event carries the agent that
    called the tool. Both are read the same way, and an event that names neither
    reports nothing rather than guessing — the run's own agent is already on
    every payload.
    """
    if source is None:
        return ""
    agent = getattr(source, "agent", None) or source
    role = getattr(agent, "role", None)
    return role.strip() if isinstance(role, str) else ""


def _text(value: Any) -> str:
    """A short, log-safe string for anything CrewAI hands us."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raw = getattr(value, "raw", None)
    if isinstance(raw, str):
        return raw
    return str(value)


def _jsonable(value: Any) -> Any:
    """Keep JSON-safe values; stringify anything else rather than failing."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def run_in_loop(loop: asyncio.AbstractEventLoop, coro: Any) -> None:
    """Hand a coroutine to the bridge's event loop from a worker thread.

    CrewAI runs synchronously on a thread; every emit therefore crosses back
    into the loop the socket lives on. Failures are logged rather than raised,
    because losing a step must never fail the turn that produced it.
    """
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:  # noqa: BLE001
        logger.exception("could not schedule a bridge emit")
