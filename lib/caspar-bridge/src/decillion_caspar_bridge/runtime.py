"""The project's agent runtime: prompts in, turns out.

This is what a Decillion prompt actually becomes. The crew creature publishes a
`crew/prompt` on the project's topic; this builds the crew from the roster that
came with it, runs it, streams the work back as it happens, and posts exactly
one answer at the end.

Two rules from the platform shape the design:

* **One writer per record.** The runtime runs the turn, so the runtime posts
  its answer — once, after the crew returns. Steps and tool calls stream from
  the event bus under the same run tag; nothing writes the same row twice.
* **A message is attributed to its agent.** Every record carries the Decillion
  program id of the agent that produced it, so a project's transcript names
  who said what rather than naming the bridge.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from .events import CrewEventForwarder
from .roster import build_roster

logger = logging.getLogger(__name__)

#: How many finished turns to keep for `crew/work`. The runtime is the record
#: of what a project's agents did, and it is bounded: a sandbox that ran for a
#: month must not hold every step it ever streamed in memory.
_WORK_HISTORY_LIMIT = 200

#: Turns are run on a worker thread (CrewAI is synchronous) with a bounded
#: number in flight, so a project that is prompted faster than it can think
#: queues rather than opening an unbounded number of model conversations.
_MAX_CONCURRENT_RUNS = 4


class CrewRuntime:
    """Runs one project's crew, and remembers what it did."""

    def __init__(self, space_id: str, send: Any) -> None:
        self._space_id = space_id
        #: `send(action, payload)` — the bridge's outbound call to the crew
        #: creature. Injected rather than imported so the runtime can be
        #: exercised without a socket.
        self._send = send
        self._history: deque[dict[str, Any]] = deque(maxlen=_WORK_HISTORY_LIMIT)
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_RUNS)
        self._active: dict[str, dict[str, Any]] = {}

    # ── the work record ──────────────────────────────────────────────────

    def work(self, agent_program_id: str = "", run_id: str = "", limit: int = 0) -> list[dict]:
        """What this project's agents have done, newest first.

        CrewAI's runs happen here, so this is the work history — there is no
        second copy kept anywhere else, which is the point: a reload reads the
        same record the runtime is still writing to.
        """
        rows = list(self._active.values()) + list(reversed(self._history))
        if agent_program_id:
            rows = [r for r in rows if r.get("agentProgramId") == agent_program_id]
        if run_id:
            rows = [r for r in rows if r.get("runId") == run_id]
        if limit and limit > 0:
            rows = rows[: int(limit)]
        return rows

    def status(self) -> dict[str, Any]:
        return {
            "spaceId": self._space_id,
            "running": len(self._active),
            "completed": len(self._history),
        }

    # ── running a turn ───────────────────────────────────────────────────

    async def handle_prompt(self, message: dict[str, Any]) -> None:
        """Run one prompt as a crew turn."""
        run_id = str(message.get("runId") or "")
        prompt = str(message.get("prompt") or "").strip()
        if not prompt:
            logger.warning("run %s carried no prompt", run_id)
            return

        thread_id = str(message.get("threadId") or "main")
        specs: list[dict[str, Any]] = list(message.get("agents") or [])
        roster = build_roster(
            specs,
            str(message.get("universalPrompt") or ""),
            dict(message.get("llmKeys") or {}),
        )
        if not roster:
            await self._post(
                "answer",
                {
                    "runId": run_id,
                    "threadId": thread_id,
                    "agentProgramId": str(message.get("agentProgramId") or ""),
                    "text": "This project has no agents that can run yet.",
                },
            )
            return

        # Which agents this turn is for: the ones it addressed, else the whole
        # team. Addressing nobody is how a project-wide prompt is expressed.
        addressed = [
            str(m.get("programId") or m)
            for m in (message.get("mentions") or [])
            if m
        ]
        target_id = str(message.get("agentProgramId") or "")
        if target_id:
            addressed.append(target_id)
        selected = [pid for pid in dict.fromkeys(addressed) if pid in roster]
        if not selected:
            selected = list(roster)

        record = {
            "runId": run_id,
            "spaceId": self._space_id,
            "threadId": thread_id,
            "agentProgramId": selected[0],
            "agents": selected,
            "prompt": prompt,
            "startedAt": time.time(),
            "status": "running",
        }
        self._active[run_id] = record

        async with self._semaphore:
            try:
                output = await self._kickoff(
                    prompt, roster, selected, specs, run_id, thread_id
                )
                record["status"] = "done"
                record["output"] = output
            except Exception as exc:  # noqa: BLE001 - a failed turn is reported, not raised
                logger.exception("run %s failed", run_id)
                record["status"] = "failed"
                record["error"] = str(exc)
                await self._post(
                    "answer",
                    {
                        "runId": run_id,
                        "threadId": thread_id,
                        "agentProgramId": selected[0],
                        "text": f"That run could not be completed: {exc}",
                    },
                )
            finally:
                record["finishedAt"] = time.time()
                self._active.pop(run_id, None)
                self._history.append(record)

    async def _kickoff(
        self,
        prompt: str,
        roster: dict[str, Any],
        selected: list[str],
        specs: list[dict[str, Any]],
        run_id: str,
        thread_id: str,
    ) -> str:
        from crewai import Crew, Process, Task

        by_id = {str(s.get("programId")): s for s in specs}
        tasks = []
        for program_id in selected:
            spec = by_id.get(program_id, {})
            tasks.append(
                Task(
                    description=prompt,
                    expected_output=str(
                        spec.get("goal")
                        or "A complete, useful answer for the project."
                    ),
                    agent=roster[program_id],
                )
            )

        crew = Crew(
            agents=[roster[pid] for pid in selected],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

        loop = asyncio.get_running_loop()

        def emit(kind: str, payload: dict[str, Any]) -> None:
            payload.setdefault("threadId", thread_id)
            asyncio.run_coroutine_threadsafe(self._post(kind, payload), loop)

        forwarder = CrewEventForwarder(emit, run_id, selected[0])
        forwarder.register()

        # CrewAI is synchronous; running it on the loop's executor keeps the
        # socket responsive, which is what lets a second prompt arrive (and a
        # step be streamed) while this turn is still thinking.
        result = await loop.run_in_executor(None, crew.kickoff)
        text = getattr(result, "raw", None) or str(result)

        # The one answer this turn writes. Steps already streamed under the
        # same run tag; nothing else posts an answer for this run.
        await self._post(
            "answer",
            {
                "runId": run_id,
                "threadId": thread_id,
                "agentProgramId": selected[0],
                "agentName": str(by_id.get(selected[0], {}).get("name") or ""),
                "text": text,
            },
        )
        return text

    # ── outbound ─────────────────────────────────────────────────────────

    async def _post(self, kind: str, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body["kind"] = kind
        body["spaceId"] = self._space_id
        body.setdefault("createdAt", time.time() * 1000)
        try:
            await self._send("crew/message", body)
        except Exception:  # noqa: BLE001 - the turn survives a lost step
            logger.exception("could not post a %s for run %s", kind, payload.get("runId"))
