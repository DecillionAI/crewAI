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
import hashlib
import logging
import re
import threading
import time
from collections import deque
from typing import Any

from .events import CrewEventForwarder
from .roster import as_manager, build_roster
from .state import read_json, state_dir, write_json_atomically
from .tools import build_tools

logger = logging.getLogger(__name__)

#: How many finished turns to keep for `crew/work`. The runtime is the record
#: of what a project's agents did, and it is bounded: a sandbox that ran for a
#: month must not hold every step it ever streamed in memory.
_WORK_HISTORY_LIMIT = 200

#: Turns are run on a worker thread (CrewAI is synchronous) with a bounded
#: number in flight, so a project that is prompted faster than it can think
#: queues rather than opening an unbounded number of model conversations.
_MAX_CONCURRENT_RUNS = 4

_MAX_HANDOFF_DEPTH = 6
_MAX_HANDOFFS_PER_ANSWER = 4


def _prose_for_mentions(text: str) -> str:
    """Remove syntax where an ``@word`` is data rather than an addressee."""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", " ", text)
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", " ", text)
    return text


def _courtesy_only(text: str, start: int, end: int) -> bool:
    """Whether the sentence merely credits/thanks the mentioned teammate."""
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("\n", 0, start))
    stops = [i for i in (text.find(".", end), text.find("!", end), text.find("\n", end)) if i >= 0]
    right = min(stops) if stops else len(text)
    sentence = text[left + 1 : right].lower()
    courtesy = re.search(r"\b(thanks?|thank you|credit|kudos|great work|well done)\b", sentence)
    instruction = re.search(
        r"\b(please|need|must|should|can you|could you|review|research|build|write|fix|create|"
        r"investigate|verify|test|finish|continue|implement|analy[sz]e|take over|handle)\b",
        sentence,
    )
    return courtesy is not None and instruction is None


def _handoff_targets(
    text: str,
    specs: list[dict[str, Any]],
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve ordered, explicit agent handles from an answer without guessing."""
    prose = _prose_for_mentions(text)
    by_handle: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        handle = str(spec.get("username") or "").strip().lstrip("@").lower()
        if handle:
            by_handle.setdefault(handle, []).append(spec)
    skipped = excluded or set()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?<![\w@])@([A-Za-z0-9][A-Za-z0-9_.-]*)", prose):
        candidates = by_handle.get(match.group(1).lower(), [])
        # A loose or ambiguous name is not authorization to spend money.
        if len(candidates) != 1 or _courtesy_only(prose, match.start(), match.end()):
            continue
        spec = candidates[0]
        program_id = str(spec.get("programId") or "")
        if not program_id or program_id in skipped or program_id in seen:
            continue
        seen.add(program_id)
        out.append(spec)
        if len(out) >= _MAX_HANDOFFS_PER_ANSWER:
            break
    return out


class _OrderedRunEmitter:
    """Serialize one run's work events before its answer/terminal event."""

    def __init__(self, runtime: "CrewRuntime", run_id: str) -> None:
        self._runtime = runtime
        self._run_id = run_id
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        self._seq = 0
        self._lock = threading.Lock()
        self._worker = asyncio.create_task(self._run())

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            seq = self._seq
        body = dict(payload)
        body["seq"] = seq
        body["eventId"] = f"{self._run_id}:{seq}"
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, body))

    async def flush(self) -> None:
        # Let callbacks scheduled from the CrewAI worker thread enqueue before
        # observing the queue's unfinished count.
        await asyncio.sleep(0)
        await self._queue.join()

    async def close(self) -> None:
        await self.flush()
        await self._queue.put(None)
        await self._worker

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            kind, payload = item
            try:
                await self._runtime._post_reliably(kind, payload)
            finally:
                self._queue.task_done()


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


def _lead_of(specs: list[dict[str, Any]], roster: dict[str, Any]) -> str:
    """The project's lead agent, if it has one and it can actually run.

    The crew creature marks it (`lead: true` on the roster entry, decided from
    the agent's `@lead` handle), so this is a lookup rather than a second place
    that knows the naming convention. An agent that failed to build is not in
    the roster, and a lead that cannot run must not silently swallow the turn.
    """
    for spec in specs:
        if spec.get("lead") is True:
            program_id = str(spec.get("programId") or "")
            if program_id in roster:
                return program_id
    return ""


def _asking_agent(message: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, str]:
    """Who a question from this turn is attributed to.

    A question is a thing an agent SAYS, so it needs an author — and the tools
    are built before the crew is, because the crew is built WITH them, so the
    agent that will actually run cannot be asked yet. The addressed agent is the
    right answer whenever there is one; a prompt addressed to the project as a
    whole is the lead's, and a project with no lead falls back to the first
    agent on it. An empty author is refused by the creature rather than posted
    as a message from nobody.
    """
    program_id = str(message.get("agentProgramId") or "").strip()
    name = ""
    by_id = {str(s.get("programId") or ""): s for s in specs}
    if not program_id:
        for spec in specs:
            if spec.get("lead") is True and spec.get("programId"):
                program_id = str(spec["programId"])
                break
    if not program_id and specs:
        program_id = str(specs[0].get("programId") or "")
    if spec := by_id.get(program_id):
        name = str(spec.get("name") or "")
    return {"agentProgramId": program_id, "agentName": name}


class CrewRuntime:
    """Runs one project's crew, and remembers what it did."""

    def __init__(
        self,
        space_id: str,
        send: Any,
        llm_proxy: Any = None,
        call: Any = None,
        await_result: Any = None,
        outbox: Any = None,
    ) -> None:
        self._space_id = space_id
        #: The platform's model proxy, if this runtime has one. Agents are
        #: pointed at it instead of at a provider, so no key is in this
        #: sandbox and the platform counts the tokens itself.
        self._llm_proxy = llm_proxy
        #: `call(action, payload)` — a REQUEST/RESPONSE call to a creature, as
        #: opposed to `send`, which only posts. It is how an agent reaches the
        #: project's Caspar tools: the tool is a creature on the node, and this
        #: is the only channel out of the sandbox.
        self._call = call
        #: `await_result(correlationId, timeout)` — wait for something published
        #: under an id this process did not mint. A question's answer comes from
        #: a person, long after the call that asked it returned, so the creature
        #: names the id up front and this is what waits on it.
        self._await_result = await_result
        #: `send(action, payload)` — the bridge's outbound call to the crew
        #: creature. Injected rather than imported so the runtime can be
        #: exercised without a socket.
        self._send = send
        #: The durable outbox, when this runtime has one. Every run event goes
        #: through it: persisted to the project's own disk first, delivered
        #: after, retried until the node accepts it. Without one the runtime
        #: still works and still retries, but only in memory — which is what a
        #: unit test wants and what a sandbox must never rely on.
        self._outbox = outbox
        self._history: deque[dict[str, Any]] = deque(maxlen=_WORK_HISTORY_LIMIT)
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_RUNS)
        self._active: dict[str, dict[str, Any]] = {}
        #: What each finished run was, on disk. `crew/work` used to answer from
        #: a bounded in-memory deque alone, so a sandbox that slept — which is
        #: every sandbox, after five idle minutes — came back having forgotten
        #: everything its agents had done.
        self._work_dir = state_dir("work")
        self._load_work_record()
        #: The loop the socket lives on, captured when the first turn starts.
        #: CrewAI runs on worker threads, so anything they emit has to be handed
        #: back across this.
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── the work record ──────────────────────────────────────────────────

    def _load_work_record(self) -> None:
        """Reload what previous processes recorded, oldest first."""
        rows: list[dict[str, Any]] = []
        for path in sorted(self._work_dir.glob("*.json")):
            row = read_json(path)
            if isinstance(row, dict) and row.get("runId"):
                rows.append(row)
        rows.sort(key=lambda row: float(row.get("startedAt") or 0))
        for row in rows[-_WORK_HISTORY_LIMIT:]:
            self._history.append(row)
        # Trim what the deque could not keep, so the directory stays the same
        # size as the record it backs rather than growing for the life of the
        # project's volume.
        for path in sorted(self._work_dir.glob("*.json"))[:-_WORK_HISTORY_LIMIT]:
            try:
                path.unlink()
            except OSError:
                pass

    def _save_work_record(self, record: dict[str, Any]) -> None:
        run_id = str(record.get("runId") or "")
        if not run_id:
            return
        safe = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_")[:96]
        if safe:
            write_json_atomically(self._work_dir / f"{safe}.json", record)


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
        if not await self._acknowledge_run(message, run_id, thread_id):
            return
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
        # What the agents on this turn can actually do. Built per turn because
        # the project's tool set travels with the prompt — a tool attached a
        # second ago is usable on the very next run, exactly like an agent added
        # a second ago is on the team.
        #
        # On a WORKER THREAD, because building the catalogue can install
        # packages: doing that on the event loop would stop the socket, and with
        # it every other project message, for as long as pip takes. Normally the
        # catalogue is already warm (see `warm_catalog`) and this returns at
        # once.
        loop = asyncio.get_running_loop()
        self._loop = loop
        # The question tool is offered only when this project can actually
        # deliver a question. That is a property of the machine's grant — fixed
        # when it was provisioned — so the platform is what says whether it is
        # routable, and a project provisioned before questions existed gets
        # agents that do not reach for one.
        turn = (
            {"runId": run_id, "threadId": thread_id, **_asking_agent(message, specs)}
            if message.get("canAsk") is True
            else None
        )
        tools = await loop.run_in_executor(
            None,
            build_tools,
            list(message.get("tools") or []),
            self._call,
            loop,
            turn,
            self._await_result,
            self._acknowledge_question,
        )
        roster = build_roster(
            specs,
            str(message.get("universalPrompt") or ""),
            proxy_base_url,
            tools,
        )
        if not roster:
            owner = _asking_agent(message, specs)
            await self._post_reliably(
                "answer",
                {
                    "runId": run_id,
                    "threadId": thread_id,
                    **owner,
                    "text": "This project has no agents that can run yet.",
                },
            )
            # Nothing ran, but the turn was authorized. Report zero usage so the
            # meter closes it out now instead of leaving the payer's funds
            # reserved until the authorization's TTL expires.
            await self._post_reliably(
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
            await self._post_reliably(
                "run-terminal",
                {
                    "runId": run_id,
                    "jobId": str(message.get("jobId") or run_id),
                    "parentRunId": str(message.get("parentRunId") or ""),
                    "threadId": thread_id,
                    **owner,
                    "status": "failed",
                    "startedAt": time.time() * 1000,
                    "endedAt": time.time() * 1000,
                    "error": "this project has no runnable agents",
                },
            )
            return

        # Who this turn is for — and, more importantly, WHICH KIND of turn it
        # is. A project has two entry points and they behave differently:
        #
        #   * the LEAD (`@lead`) is where a whole objective goes. The turn is
        #     the crew's: the lead runs it as the crew's manager, decomposing
        #     the work and delegating to whichever teammates it needs, and one
        #     answer comes back under the lead's name.
        #   * any OTHER agent, mentioned by name, is that agent's turn alone —
        #     unchanged, and the reason both are worth having.
        #
        # Addressing nobody is a project-wide prompt, which is the lead's job
        # when there is one; a project with no lead falls back to running the
        # whole roster, as it always did.
        lead_id = _lead_of(specs, roster)
        addressed = [
            str(m.get("programId") or m)
            for m in (message.get("mentions") or [])
            if m
        ]
        target_id = str(message.get("agentProgramId") or "")
        if target_id:
            addressed.append(target_id)
        selected = [pid for pid in dict.fromkeys(addressed) if pid in roster]
        crew_turn = bool(lead_id) and (not selected or lead_id in selected)
        if crew_turn:
            # The lead owns the record and writes the answer; the teammates it
            # delegates to report their work as steps under the same run.
            selected = [lead_id]
        elif not selected:
            selected = list(roster)

        started_at = time.time()
        record = {
            "runId": run_id,
            "jobId": str(message.get("jobId") or run_id),
            "parentRunId": str(message.get("parentRunId") or ""),
            "spaceId": self._space_id,
            "threadId": thread_id,
            "agentProgramId": selected[0],
            "agentName": str(next((s.get("name") for s in specs if str(s.get("programId")) == selected[0]), "") or ""),
            "agents": selected,
            "prompt": prompt,
            "startedAt": started_at,
            "status": "running",
        }
        self._active[run_id] = record

        usage: dict[str, Any] = {"promptTokens": 0, "completionTokens": 0}
        heartbeat = asyncio.create_task(self._heartbeat(record))
        async with self._semaphore:
            try:
                output = await self._kickoff(
                    prompt,
                    roster,
                    selected,
                    specs,
                    run_id,
                    thread_id,
                    usage,
                    lead_id if crew_turn else "",
                )
                record["status"] = "done"
                record["output"] = output
                await self._note_mentions(specs, selected, record, output)
            except Exception as exc:  # noqa: BLE001 - a failed turn is reported, not raised
                logger.exception("run %s failed", run_id)
                record["status"] = "failed"
                record["error"] = str(exc)
                await self._post_reliably(
                    "answer",
                    {
                        "runId": run_id,
                        "threadId": thread_id,
                        "agentProgramId": selected[0],
                        "agentName": record.get("agentName") or "",
                        "text": f"That run could not be completed: {exc}",
                    },
                )
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
                finished_at = time.time()
                record["finishedAt"] = finished_at
                record["usage"] = dict(usage)
                self._active.pop(run_id, None)
                self._history.append(record)
                self._save_work_record(record)
                # What the run cost, reported once. The meter settles from this;
                # a lost report leaves the run's authorization to expire rather
                # than overcharging, which is why it is sent last and separately
                # from the answer.
                await self._post_reliably(
                    "usage",
                    {
                        "runId": run_id,
                        "threadId": thread_id,
                        "agentProgramId": record.get("agentProgramId") or "",
                        "runtimeMs": int(max(0.0, finished_at - started_at) * 1000),
                        "promptTokens": int(usage.get("promptTokens") or 0),
                        "completionTokens": int(usage.get("completionTokens") or 0),
                        # What each agent spent. A led turn is one run, but the
                        # work inside it is done by whichever teammates the lead
                        # delegated to, and their creators are owed for it.
                        "agents": usage.get("agents") or [],
                        "success": record.get("status") == "done",
                    },
                )
                await self._post_reliably(
                    "run-terminal",
                    {
                        "runId": run_id,
                        "jobId": record.get("jobId") or run_id,
                        "parentRunId": record.get("parentRunId") or "",
                        "threadId": thread_id,
                        "agentProgramId": record.get("agentProgramId") or "",
                        "agentName": record.get("agentName") or "",
                        "status": "succeeded" if record.get("status") == "done" else "failed",
                        "startedAt": started_at * 1000,
                        "endedAt": finished_at * 1000,
                        "error": record.get("error") or "",
                    },
                )

    async def _acknowledge_run(
        self, message: dict[str, Any], run_id: str, thread_id: str
    ) -> bool:
        """Tell the platform this runtime has the turn, and learn whether to run it.

        Two things happen on this one call, and both have to happen before the
        turn does anything slow:

        * The platform holds every prompt in a durable inbox and replays whatever
          nobody acknowledged. Without this a healthy long run would be
          redelivered each time its dispatch lease expired, and a sandbox that
          died between the publish and its first token would look exactly like
          one that was working.
        * The payer's slice is taken **here**, by the creature this reaches —
          the only program the node lets reserve against the pool. So a run the
          payer cannot fund is stopped now, before a tool is built or a model is
          called, rather than after its answer has been given away.

        Returns whether to proceed. A refusal ends the turn quietly: the platform
        has already closed the run and said why.
        """
        payload = {
            "runId": run_id,
            "jobId": str(message.get("jobId") or run_id),
            "threadId": thread_id,
            "agentProgramId": str(message.get("agentProgramId") or ""),
        }
        if self._call is None:
            # No request/response channel (a runtime under test). Record it
            # durably and carry on — the acknowledgement still lands, and the
            # reservation is still made when it does.
            await self._post_reliably("run-ack", payload)
            return True
        try:
            result = await self._call("crew/message", {**payload, "kind": "run-ack"})
        except Exception as exc:  # noqa: BLE001 - an unreachable platform is not a refusal
            logger.warning("could not acknowledge run %s: %s", run_id, exc)
            await self._post_reliably("run-ack", payload)
            return True
        if isinstance(result, dict) and result.get("stop") is True:
            logger.warning(
                "run %s was refused by the platform: %s", run_id, result.get("error")
            )
            return False
        return True

    def _acknowledge_question(self, question_id: str) -> None:
        """Confirm a person's answer reached the run that was waiting for it.

        Called from the CrewAI worker thread, so it goes through the outbox
        rather than the socket: the acknowledgement is what retires the question
        on the platform, and one that is dropped leaves a question open forever
        against a run that has already moved on.
        """
        if not question_id:
            return
        payload = {"runId": "", "questionId": question_id}
        if self._outbox is not None:
            self._outbox.post("question-ack", payload)
            return
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._post("question-ack", payload), loop)
        except Exception:  # noqa: BLE001 - the answer still stands
            logger.exception("could not acknowledge question %s", question_id)

    async def _heartbeat(self, record: dict[str, Any]) -> None:
        """Keep the durable ledger fresh while queued or inside a long model call."""
        while True:
            await asyncio.sleep(20)
            await self._post_reliably(
                "heartbeat",
                {
                    "runId": str(record.get("runId") or ""),
                    "jobId": str(record.get("jobId") or ""),
                    "parentRunId": str(record.get("parentRunId") or ""),
                    "threadId": str(record.get("threadId") or "main"),
                    "agentProgramId": str(record.get("agentProgramId") or ""),
                    "agentName": str(record.get("agentName") or ""),
                },
            )
    @staticmethod
    def _build_crew(
        prompt: str,
        roster: dict[str, Any],
        selected: list[str],
        by_id: dict[str, Any],
        lead_id: str,
        Crew: Any,
        Process: Any,
        Task: Any,
    ) -> Any:
        """The crew that runs this turn.

        Two shapes, because a project has two kinds of turn:

        **Led.** The objective goes to the crew as ONE task with no agent on
        it, and the lead is the crew's `manager_agent`. CrewAI's hierarchical
        process is exactly this: the manager plans the work, delegates each
        piece to the teammate best suited to it, and returns one result. That
        is what makes `@lead` a collaboration rather than a broadcast — the
        teammates are chosen by the lead, per step, from what the task needs.

        CrewAI requires the manager to be outside `agents` and turns its
        delegation on itself, so the teammates keep `allow_delegation=False`
        and only the lead hands work out.

        **Direct.** One task per addressed agent, run in order, each answering
        as itself. Unchanged: mentioning an agent by name is still how you
        reach that agent and nobody else.
        """
        if lead_id:
            # The manager carries NO tools, and CrewAI enforces it ("Manager
            # agent should not have tools"). The rule is right: in a
            # hierarchical crew the manager decides who does the work and the
            # teammates do it, so a manager holding tools would be a manager
            # doing the job itself — which is the one thing delegation is for.
            # Its teammates keep every tool; `as_manager` is what gives them up,
            # and what tells the lead so.
            manager = as_manager(roster[lead_id])
            teammates = [agent for pid, agent in roster.items() if pid != lead_id]
            goal = str(
                by_id.get(lead_id, {}).get("goal")
                or "A complete, useful result for the project."
            )
            if teammates:
                return Crew(
                    agents=teammates,
                    # No `agent=`: in a hierarchical crew the manager decides
                    # who executes, and pinning one here would defeat that.
                    tasks=[Task(description=prompt, expected_output=goal)],
                    process=Process.hierarchical,
                    manager_agent=manager,
                    verbose=False,
                )
            # A lead with nobody to lead is just an agent, and it does the work
            # itself — so this one runs WITH its tools. Running it as a
            # hierarchical crew with an empty roster would fail validation, and
            # the person asked for the work either way.
            solo = roster[lead_id]
            return Crew(
                agents=[solo],
                tasks=[Task(description=prompt, expected_output=goal, agent=solo)],
                process=Process.sequential,
                verbose=False,
            )

        tasks = [
            Task(
                description=prompt,
                expected_output=str(
                    by_id.get(program_id, {}).get("goal")
                    or "A complete, useful answer for the project."
                ),
                agent=roster[program_id],
            )
            for program_id in selected
        ]
        return Crew(
            agents=[roster[pid] for pid in selected],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
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
        lead_id: str = "",
    ) -> str:
        from crewai import Crew, Process, Task

        by_id = {str(s.get("programId")): s for s in specs}
        crew = self._build_crew(prompt, roster, selected, by_id, lead_id, Crew, Process, Task)

        loop = asyncio.get_running_loop()

        ordered = _OrderedRunEmitter(self, run_id)

        def emit(kind: str, payload: dict[str, Any]) -> None:
            payload.setdefault("threadId", thread_id)
            ordered.emit(kind, payload)

        actors = {
            id(agent): (pid, str(by_id.get(pid, {}).get("name") or ""))
            for pid, agent in roster.items()
        }
        # CrewAI addresses a coworker by ROLE, and the project's chat addresses
        # one by @handle. Both spellings map to the same agent here so a
        # delegation can be posted the way the project talks.
        handles: dict[str, str] = {}
        for pid, spec in by_id.items():
            handle = str(spec.get("username") or "").lstrip("@")
            if not handle:
                continue
            for spelling in (spec.get("role"), spec.get("name"), handle):
                if spelling:
                    handles[str(spelling).strip().lower()] = handle
        sources = [crew, *getattr(crew, "tasks", []), *roster.values()]
        forwarder = CrewEventForwarder(
            emit,
            run_id,
            selected[0],
            str(by_id.get(selected[0], {}).get("name") or ""),
            sources=sources,
            actors=actors,
            handles=handles,
        )
        forwarder.register()

        # CrewAI is synchronous; running it on the loop's executor keeps the
        # socket responsive, which is what lets a second prompt arrive (and a
        # step be streamed) while this turn is still thinking.
        try:
            result = await loop.run_in_executor(None, crew.kickoff)
        finally:
            # Event handlers are global inside CrewAI. Tear this run's handlers
            # down before another turn starts, then drain every event already
            # emitted so the answer cannot overtake its final completed step.
            # Read the per-agent split BEFORE tearing down: it is what makes a
            # led turn pay every agent that worked on it, not only the lead
            # whose name the run carries.
            usage["agents"] = [
                {"programId": pid, "runtimeMs": ms}
                for pid, ms in forwarder.agent_runtime_ms().items()
            ]
            forwarder.unregister()
            await ordered.close()
        text = getattr(result, "raw", None) or str(result)
        # Token counts come from the crew that just ran. They are written into
        # the caller's dict rather than returned, so a turn that raises after
        # the model was called still reports what it burned.
        usage.update(_token_usage(result, crew))

        # The one answer this turn writes. Steps already streamed under the
        # same run tag; nothing else posts an answer for this run.
        await self._post_reliably(
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

    async def _note_mentions(
        self,
        specs: list[dict[str, Any]],
        selected: list[str],
        record: dict[str, Any],
        answer: str,
    ) -> None:
        """Record the teammates an answer named — without starting any of them.

        A mention written by an AGENT is a reference, not a request. Agents
        collaborate inside the crew: the lead is CrewAI's `manager_agent` and
        CrewAI's own delegation picks who does each step, all within the one
        turn the person asked for. So an answer that names a colleague is
        describing work that has already been shared, and re-launching that
        colleague as a fresh run would do it a second time.

        Mentions written by a PERSON are the opposite and still start agents:
        the app sends them as the turn's seeds. That asymmetry is the whole
        rule — a person addresses an agent, agents address each other inside
        the crew.

        This used to launch a separately billed child run per mention, which is
        why the runtime needed a delegated billing pool and why an answer with
        no pool attached posted "this run has no delegated billing pool for a
        server-side hand-off" instead of collaborating.
        """
        targets = _handoff_targets(answer, specs, set(selected))
        if not targets:
            return
        named = ", ".join(
            "@" + str(t.get("username") or t.get("name") or t.get("programId") or "").lstrip("@")
            for t in targets
        )
        await self._post_reliably(
            "step",
            {
                "runId": str(record.get("runId") or ""),
                "jobId": str(record.get("jobId") or ""),
                "threadId": str(record.get("threadId") or "main"),
                "agentProgramId": str(record.get("agentProgramId") or ""),
                "agentName": str(record.get("agentName") or ""),
                "status": "completed",
                "text": f"Referred to {named} — the crew works this out together, "
                        "so nothing was started separately.",
            },
        )

    async def _handoff_notice(self, record: dict[str, Any], text: str) -> None:
        await self._post_reliably(
            "step",
            {
                "runId": str(record.get("runId") or ""),
                "jobId": str(record.get("jobId") or ""),
                "parentRunId": str(record.get("parentRunId") or ""),
                "threadId": str(record.get("threadId") or "main"),
                "agentProgramId": str(record.get("agentProgramId") or ""),
                "agentName": str(record.get("agentName") or ""),
                "status": "failed",
                "text": text,
            },
        )

    # ── outbound ─────────────────────────────────────────────────────────

    async def _post(self, kind: str, payload: dict[str, Any]) -> bool:
        body = dict(payload)
        body["kind"] = kind
        body["spaceId"] = self._space_id
        body.setdefault("createdAt", time.time() * 1000)
        try:
            await self._send("crew/message", body)
            return True
        except Exception:  # noqa: BLE001 - the turn survives a lost step
            logger.exception("could not post a %s for run %s", kind, payload.get("runId"))
            return False

    async def _post_reliably(self, kind: str, payload: dict[str, Any]) -> bool:
        """Record a run event durably, then let the outbox deliver it.

        With an outbox this returns as soon as the event is on the project's own
        disk, which is the point: delivery is then somebody else's problem and it
        is retried until the node accepts it, across reconnects, restarts and the
        sandbox being replaced. Ordering is preserved because the outbox drains
        in the order events were produced.

        Without one — a unit test, a runtime exercised with no filesystem — it
        falls back to the old in-memory retries. Three attempts over three
        quarters of a second is not a delivery guarantee, which is exactly why it
        is no longer what a sandbox uses.
        """
        if self._outbox is not None:
            self._outbox.post(kind, payload)
            return True
        for attempt in range(3):
            if await self._post(kind, payload):
                return True
            if attempt < 2:
                await asyncio.sleep(0.25 * (attempt + 1))
        return False
