"""The tool server: one per project, running inside its Modal sandbox.

This process used to be the project's *agent runtime* — it built a CrewAI crew
per turn, held the roster, proxied model calls, and forwarded events. All of
that now runs on Caspar as the Da Vinci engine, and what is left here is the one
thing that genuinely cannot: **running tools that need this machine.**

Two kinds of them, and no others:

* the ``crewai_tools`` catalogue, which is a large pile of vendor SDKs that has
  no business inside a WASM creature; and
* the project's own workspace — its files, its shell, its checked-out
  repositories — which exist on this machine's volume and nowhere else.

Everything else an agent can do (remembering, delegating, asking a person,
calling a creature tool, calling an MCP server) happens on the node, which is
why a turn that does not touch a file no longer wakes this machine at all.

## What this process is NOT allowed to do

It holds no Caspar key. Its whole authority is a bearer token scoped to one
topic, and the grant names exactly two routes: ``crew/tool`` and ``crew/status``.
``llm/chat`` is deliberately not among them — this process makes no model calls,
so a container escape reaches no provider credential. That is a structural
property, not a policy: the node refuses a route the grant does not name.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from .tools import build_catalog, run_tool, tool_manifest

logger = logging.getLogger(__name__)

#: How many tools may run at once. A sandbox is one small machine and a
#: catalogue tool can be a whole vendor SDK; letting a project run twenty at
#: once is how one turn takes the machine away from every other.
_MAX_CONCURRENT_TOOLS = 4

#: A tool that has not answered by here is reported as timed out rather than
#: left to the node's frame lease. The node's lease is the backstop; this is the
#: message that actually says what happened.
_TOOL_TIMEOUT_SECS = 600.0


class ToolServer:
    """Runs the tools this project's agents ask for, and reports what they did.

    One instance per process. It owns no conversation state and no roster: a
    request names a tool and its arguments, and the answer goes back against the
    call id it arrived with. Restarting it loses nothing, which is why the
    sandbox's entrypoint can simply loop.
    """

    def __init__(
        self,
        send: Callable[[str, dict], Awaitable[dict]],
        space_id: str,
        runtime_ref: str = "",
    ) -> None:
        self._send = send
        self._space_id = space_id
        self._runtime_ref = runtime_ref
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TOOLS)
        self._running: dict[str, asyncio.Task] = {}

    # ── Announcing ───────────────────────────────────────────────────────

    async def announce(self) -> None:
        """Tell the node this machine is ready, and what it can run.

        This is what replays a project's backlog. A tool call made while the
        machine was asleep is held on the node as a suspended frame, and the
        node republishes every one of them when this lands — so a woken sandbox
        is handed its work with no client present and nobody is asked to try
        again.

        The catalogue travels with the announcement because the node needs to
        know which tools exist here before it can offer them to an agent, and
        asking for it separately would be a round trip on the critical path of
        every first prompt.
        """
        manifest = tool_manifest()
        logger.info("announcing %d tools for %s", len(manifest), self._space_id)
        await self._send(
            "crew/bridge",
            {
                "fn": "announce",
                "spaceId": self._space_id,
                "tools": manifest,
                "ref": self._runtime_ref,
            },
        )

    # ── Running ──────────────────────────────────────────────────────────

    async def on_request(self, request: dict[str, Any]) -> None:
        """Handle one ``tool/invoke`` packet pushed onto this project's topic.

        Returns as soon as the work is scheduled. The node is not waiting on
        this call — it is waiting on a durable frame — so blocking the socket
        while a tool runs would only stop the next request arriving.
        """
        call_id = str(request.get("callId") or "")
        if not call_id:
            logger.warning("ignoring a tool request with no callId")
            return
        if call_id in self._running:
            # At-least-once delivery: the node republishes anything it has not
            # had an answer for, including things that are simply still running.
            logger.info("tool call %s is already running", call_id)
            return

        task = asyncio.create_task(self._run(call_id, request))
        self._running[call_id] = task
        task.add_done_callback(lambda _t, cid=call_id: self._running.pop(cid, None))

    async def _run(self, call_id: str, request: dict[str, Any]) -> None:
        tool = str(request.get("tool") or "")
        args = request.get("args") or {}
        started = time.monotonic()

        async with self._semaphore:
            try:
                # `run_tool` is synchronous — a catalogue tool is ordinary
                # blocking Python — so it goes to a worker thread. Running it on
                # the event loop would stall every other tool and the socket
                # with it.
                result = await asyncio.wait_for(
                    asyncio.to_thread(run_tool, tool, args),
                    timeout=_TOOL_TIMEOUT_SECS,
                )
                outcome = {"ok": True, "result": result}
            except asyncio.TimeoutError:
                outcome = {
                    "ok": False,
                    "error": (
                        f"{tool} did not finish within {int(_TOOL_TIMEOUT_SECS)} seconds "
                        "and was stopped"
                    ),
                }
            except BaseException as exc:  # noqa: BLE001
                # A tool failure is an ANSWER, not a crash. The agent reads it
                # and reasons about it; ending the turn over a vendor's bad day
                # would be the wrong trade.
                logger.exception("tool %s failed", tool)
                outcome = {"ok": False, "error": f"{tool} could not be run: {exc or type(exc).__name__}"}

        outcome["callId"] = call_id
        outcome["durationMs"] = int((time.monotonic() - started) * 1000)
        await self._report(outcome)

    async def _report(self, outcome: dict[str, Any]) -> None:
        """Send one result back, letting the outbox retry it.

        The node answers a tool result with an acknowledgement; anything else
        means it has not landed, and the outbox keeps it until it does. A result
        this process drops is a run that waits out its frame lease for no reason.
        """
        try:
            await self._send("crew/bridge", {"fn": "result", **outcome})
        except Exception:  # noqa: BLE001
            logger.exception("could not report tool %s", outcome.get("callId"))

    # ── Readiness ────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Say the machine is still here.

        The node treats a report older than a few minutes as "this server went
        away without saying so" and wakes the machine on the next tool call. A
        heartbeat is much cheaper than a wake.
        """
        await self._send("crew/status", {"fn": "toolServer", "spaceId": self._space_id, "at": int(time.time() * 1000)})

    def warm(self) -> int:
        """Build the catalogue now, off the hot path.

        Saying yes to the tool installers means the first build can take
        minutes, and doing that lazily would spend them inside somebody's first
        prompt — a project that looks hung at exactly the moment somebody is
        watching it.
        """
        return len(build_catalog())
