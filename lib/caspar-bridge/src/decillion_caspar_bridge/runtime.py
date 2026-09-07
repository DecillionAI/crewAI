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
* **A turn that ran gets settled.** The runtime is the only party that knows
  what a run actually used, so every turn ends with one `usage` report to the
  crew creature — the node's registered settlement meter — carrying the run's
  duration and its token counts. It is posted whether the turn succeeded or
  failed, because a failed turn still burned tokens. The bridge computes no
  prices and holds no billing identifiers: it reports observations, and the
  meter prices them against the quote the run was authorized under.
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


def _token_usage(result: Any, crew: Any) -> dict[str, int]:
    """Prompt/completion tokens for a finished crew run.

    CrewAI exposes them on the output and again on the crew, and the attribute
    names have moved between versions, so every shape is tried and an unknown
    one reports zero rather than raising: a run must never fail to be recorded
    because its token counter was renamed.
    """
    for source in (getattr(result, "token_usage", None), getattr(crew, "usage_metrics", None)):
        if source is None:
            continue
        prompt = _first_int(source, ("prompt_tokens", "promptTokens"))
        completion = _first_int(source, ("completion_tokens", "completionTokens"))
        if prompt or completion:
            return {"promptTokens": prompt, "completionTokens": completion}
    return {"promptTokens": 0, "completionTokens": 0}


def _first_int(source: Any, names: tuple[str, ...]) -> int:
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


class CrewRuntime:
    """Runs one project's crew, and remembers what it did."""

    def __init__(self, space_id: str, send: Any, llm_proxy: Any = None) -> None:
        self._space_id = space_id
        #: The platform's model proxy, if this runtime has one. Agents are
        #: pointed at it instead of at a provider, so no key is in this
        #: sandbox and the platform counts the tokens itself.
        self._llm_proxy = llm_proxy
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
        # Bind each agent's model to its Decillion provider before the crew is
        # built: LiteLLM sends the proxy only a model string, and the creature
        # has to know whose key to spend. The provider ids are the platform's
        # one vocabulary — never guessed from a model name.
        proxy_base_url = ""
        if self._llm_proxy is not None:
            proxy = dict(message.get("llmProxy") or {})
            if proxy.get("action"):
                self._llm_proxy.set_action(str(proxy["action"]))
            for spec in specs:
                spec_llm = spec.get("llm") or {}
                self._llm_proxy.bind_model(
                    str(spec_llm.get("model") or ""),
                    str(spec_llm.get("provider") or "").strip().lower(),
                )
            proxy_base_url = self._llm_proxy.base_url
        roster = build_roster(
            specs,
            str(message.get("universalPrompt") or ""),
            proxy_base_url,
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
            # Nothing ran, but the turn was authorized. Report zero usage so the
            # meter closes it out now instead of leaving the payer's funds
            # reserved until the authorization's TTL expires.
            await self._post(
                "usage",
                {
                    "runId": run_id,
                    "threadId": thread_id,
                    "agentProgramId": str(message.get("agentProgramId") or ""),
                    "runtimeMs": 0,
                    "promptTokens": 0,
                    "completionTokens": 0,
                    "success": False,
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

        started_at = time.time()
        record = {
            "runId": run_id,
            "spaceId": self._space_id,
            "threadId": thread_id,
            "agentProgramId": selected[0],
            "agents": selected,
            "prompt": prompt,
            "startedAt": started_at,
            "status": "running",
        }
        self._active[run_id] = record

        usage: dict[str, Any] = {"promptTokens": 0, "completionTokens": 0}
        async with self._semaphore:
            try:
                output = await self._kickoff(
                    prompt, roster, selected, specs, run_id, thread_id, usage
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
                finished_at = time.time()
                record["finishedAt"] = finished_at
                record["usage"] = dict(usage)
                self._active.pop(run_id, None)
                self._history.append(record)
                # What the run cost, reported once. The meter settles from this;
                # a lost report leaves the run's authorization to expire rather
                # than overcharging, which is why it is sent last and separately
                # from the answer.
                await self._post(
                    "usage",
                    {
                        "runId": run_id,
                        "threadId": thread_id,
                        "agentProgramId": record.get("agentProgramId") or "",
                        "runtimeMs": int(max(0.0, finished_at - started_at) * 1000),
                        "promptTokens": int(usage.get("promptTokens") or 0),
                        "completionTokens": int(usage.get("completionTokens") or 0),
                        "success": record.get("status") == "done",
                    },
                )

    async def _kickoff(
        self,
        prompt: str,
        roster: dict[str, Any],
        selected: list[str],
        specs: list[dict[str, Any]],
        run_id: str,
        thread_id: str,
        usage: dict[str, Any],
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
        # Token counts come from the crew that just ran. They are written into
        # the caller's dict rather than returned, so a turn that raises after
        # the model was called still reports what it burned.
        usage.update(_token_usage(result, crew))

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
