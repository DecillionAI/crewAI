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

    def __init__(self, emit: Emitter, run_id: str, agent_program_id: str) -> None:
        self._emit = emit
        self._run_id = run_id
        self._agent_program_id = agent_program_id
        self._registered: list[Any] = []

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

        @bus.on(TaskStartedEvent)
        def _task_started(_source: Any, event: Any) -> None:
            self._step("started", getattr(event, "task", None))

        @bus.on(TaskCompletedEvent)
        def _task_completed(_source: Any, event: Any) -> None:
            self._step("completed", getattr(event, "task", None), getattr(event, "output", None))

        @bus.on(TaskFailedEvent)
        def _task_failed(_source: Any, event: Any) -> None:
            self._step("failed", getattr(event, "task", None), getattr(event, "error", None))

        @bus.on(ToolUsageStartedEvent)
        def _tool_started(_source: Any, event: Any) -> None:
            self._tool("started", event)

        @bus.on(ToolUsageFinishedEvent)
        def _tool_finished(_source: Any, event: Any) -> None:
            self._tool("finished", event)

        @bus.on(ToolUsageErrorEvent)
        def _tool_error(_source: Any, event: Any) -> None:
            self._tool("error", event)

        self._registered = [
            _task_started,
            _task_completed,
            _task_failed,
            _tool_started,
            _tool_finished,
            _tool_error,
        ]

    # ── emitters ─────────────────────────────────────────────────────────

    def _step(self, state: str, task: Any, detail: Any = None) -> None:
        description = _text(getattr(task, "description", "")) if task else ""
        self._emit(
            "step",
            {
                "runId": self._run_id,
                "agentProgramId": self._agent_program_id,
                "status": state,
                "text": description,
                "data": {"detail": _text(detail)} if detail is not None else None,
            },
        )

    def _tool(self, state: str, event: Any) -> None:
        self._emit(
            "toolcall",
            {
                "runId": self._run_id,
                "agentProgramId": self._agent_program_id,
                "status": state,
                "toolName": getattr(event, "tool_name", "") or "",
                "toolArgs": _jsonable(getattr(event, "tool_args", None)),
                "toolResult": _text(getattr(event, "output", None)),
                "text": getattr(event, "tool_name", "") or "",
            },
        )


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
