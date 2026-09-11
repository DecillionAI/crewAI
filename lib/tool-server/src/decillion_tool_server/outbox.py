"""Run events that must reach the node, even if this process does not.

Everything a run produces — its steps, its tool calls, its answer, what it cost,
how it ended — travels to the crew creature over one socket. That socket belongs
to a sandbox which sleeps, restarts and is replaced, and the events are the only
record of work somebody has already paid for.

The previous delivery policy was three attempts across roughly three quarters of
a second, held entirely in memory. A node restart, a reconnect, or the sandbox
being terminated mid-turn silently dropped whatever was in flight — including the
usage report, which is the event the run is BILLED from. Losing that is not a
missing log line, it is work delivered for free.

So an event is written to the project's own disk before it is sent, and it stays
there until the node has accepted it:

    produced ──► persisted to /data ──► sent ──► accepted ──► deleted

Retries are unbounded and backed off, replay happens on startup, and ordering is
preserved by a monotonic sequence, so a turn's answer never overtakes the steps
that led to it. Delivery is at-least-once and the creature is idempotent by
`eventId`, sequence number and run id — which is the right way round: a
duplicated step is a no-op, a lost settlement is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from .state import read_json, state_dir, write_json_atomically

logger = logging.getLogger(__name__)

#: Backoff between delivery attempts, in seconds. It caps rather than growing
#: without bound: a sandbox can outlive an outage by days, and an event that has
#: been waiting an hour should still go out within a minute of the node coming
#: back.
_BACKOFF_START = 0.5
_BACKOFF_CAP = 30.0

#: How many times one event is retried at the head of the queue before later
#: events are allowed past it. High enough that any ordinary outage — a
#: reconnect, a node restart — is ridden out in order; finite so one permanently
#: refused event cannot wedge a project's whole transcript behind it.
_HEAD_OF_LINE_ATTEMPTS = 12

#: Events whose delivery must be CONFIRMED by the creature rather than merely
#: accepted by the gateway. `usage` is the settlement report: the gateway's
#: acknowledgement says the packet arrived, and says nothing at all about whether
#: the run was actually billed.
_CONFIRMED_KINDS = frozenset({"usage"})

#: Where an event goes when nothing says otherwise — the meter, which is where
#: every event this queue was built for went. Kept as the default so a record
#: left on a project's volume by an older process still delivers where it was
#: addressed when it was written.
_DEFAULT_ACTION = "crew/message"


def _now_ms() -> float:
    return time.time() * 1000


class Outbox:
    """A durable, ordered, at-least-once queue of one project's run events."""

    def __init__(
        self,
        space_id: str,
        send: Callable[[str, dict], Awaitable[dict]],
        call: Callable[[str, dict], Awaitable[dict]] | None = None,
        *,
        directory: Path | None = None,
    ) -> None:
        self._space_id = space_id
        self._send = send
        #: Request/response to the creature, used for the events whose ANSWER
        #: matters (see `_CONFIRMED_KINDS`). Absent in tests that only assert
        #: persistence.
        self._call = call
        self._dir = directory or state_dir("outbox")
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._seq = self._highest_sequence()
        self._closing = False
        #: Set while the queue is empty and nothing is being delivered — what
        #: `drain()` waits on.
        self._idle = asyncio.Event()
        self._idle.set()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin delivering, replaying anything a previous process left behind."""
        if self._worker is not None:
            return
        recovered = self.replay()
        if recovered:
            logger.info("replaying %d undelivered run events", recovered)
        self._worker = asyncio.create_task(self._drain())

    async def close(self) -> None:
        self._closing = True
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    def replay(self) -> int:
        """Re-enqueue every event still on disk, oldest first."""
        names = sorted(path.name for path in self._dir.glob("*.json"))
        for name in names:
            self._queue.put_nowait(name)
        if names:
            self._idle.clear()
        return len(names)

    # ── producing ────────────────────────────────────────────────────────

    def post(self, kind: str, payload: dict[str, Any], *, action: str = _DEFAULT_ACTION) -> str:
        """Persist one event and queue it. Returns its stable event id.

        Synchronous on purpose: the write has to happen before the caller can
        believe the event exists, and the caller is often a CrewAI worker thread
        that has no loop of its own.

        `action` is the creature this event is FOR, recorded with it. It used to
        be implicit — everything this process produced was a run event for the
        meter — and that stopped being true when the agents moved to the node:
        what a sandbox produces now is a tool result for `crew/bridge`. A queue
        that knew only one destination could not carry it.
        """
        self._seq += 1
        body = dict(payload)
        body["kind"] = kind
        body["spaceId"] = self._space_id
        body.setdefault("createdAt", _now_ms())
        # A stable id the creature can dedupe on, distinct from the per-run
        # sequence the ordered emitter already stamps: this one identifies the
        # DELIVERY, so a replayed event is recognisably the same one.
        event_id = str(body.get("eventId") or "") or uuid.uuid4().hex
        body["eventId"] = event_id
        name = f"{self._seq:012d}-{event_id}.json"
        record = {
            "name": name,
            "kind": kind,
            "action": action,
            "eventId": event_id,
            "attempts": 0,
            "queuedAt": _now_ms(),
            "payload": body,
        }
        if not write_json_atomically(self._dir / name, record):
            # The disk refused it. Still try to deliver — an event in memory is
            # worse than one on disk and better than none at all.
            logger.error("could not persist a %s event; delivering without it", kind)
        self._idle.clear()
        self._queue.put_nowait(name)
        return event_id

    async def drain(self, timeout: float | None = None) -> bool:
        """Wait until everything queued has been delivered.

        Used where ordering across a boundary matters — a turn flushing its work
        before its terminal event — and by tests. Returns whether it emptied.
        """
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def pending(self) -> int:
        return len(list(self._dir.glob("*.json")))

    @property
    def pending_tool_call_ids(self) -> set[str]:
        """Tool results already durably queued for delivery.

        A catalogue refresh can make the node replay a call after the tool has
        finished but before its queued result reaches the creature.  The tool
        server uses this set when it starts so that replay does not execute the
        side effect again while the outbox is already carrying its answer.
        """
        call_ids: set[str] = set()
        for path in self._dir.glob("*.json"):
            record = read_json(path)
            if not isinstance(record, dict) or record.get("action") != "crew/bridge":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("fn") != "result":
                continue
            call_id = str(payload.get("callId") or "")
            if call_id:
                call_ids.add(call_id)
        return call_ids

    # ── delivering ───────────────────────────────────────────────────────

    async def _drain(self) -> None:
        while not self._closing:
            name = await self._queue.get()
            try:
                await self._deliver_in_order(name)
            except asyncio.CancelledError:
                # Leave the event on disk: the next process replays it.
                self._queue.put_nowait(name)
                raise
            finally:
                self._queue.task_done()
                self._settle_idle()

    async def _deliver_in_order(self, name: str) -> None:
        """Deliver one event, holding the line until it goes.

        Retrying in PLACE rather than requeueing is what keeps a turn readable:
        the connection failures this retries are the kind that affect every
        event equally, so letting the next one past would put an answer in the
        transcript ahead of the steps that produced it.

        The one thing that must not happen is a single undeliverable event
        wedging a project forever, so after `_HEAD_OF_LINE_ATTEMPTS` it goes to
        the back of the queue instead. It is still retried, and still not lost —
        it simply stops blocking work that has nothing to do with it.
        """
        record = read_json(self._dir / name)
        if not isinstance(record, dict):
            # Gone (already delivered) or unreadable. Nothing left to send.
            return
        backoff = _BACKOFF_START
        for _ in range(_HEAD_OF_LINE_ATTEMPTS):
            if self._closing:
                return
            if await self._deliver(record):
                self._discard(name)
                return
            record["attempts"] = int(record.get("attempts") or 0) + 1
            record["lastAttemptAt"] = _now_ms()
            write_json_atomically(self._dir / name, record)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_CAP)
        logger.error(
            "a %s for run %s has not been accepted after %d attempts; "
            "letting later events past while it keeps retrying",
            record.get("kind"),
            (record.get("payload") or {}).get("runId"),
            _HEAD_OF_LINE_ATTEMPTS,
        )
        self._idle.clear()
        self._queue.put_nowait(name)

    def _settle_idle(self) -> None:
        if self._queue.empty():
            self._idle.set()

    def _discard(self, name: str) -> None:
        with contextlib.suppress(OSError):
            (self._dir / name).unlink()

    async def _deliver(self, record: dict[str, Any]) -> bool:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return True
        kind = str(record.get("kind") or "")
        # A record written by an OLDER process names no action: it can only have
        # been a run event for the meter, which is where those went.
        action = str(record.get("action") or _DEFAULT_ACTION)
        try:
            if kind in _CONFIRMED_KINDS and self._call is not None:
                # The creature's OWN verdict, not the gateway's acknowledgement.
                # A settlement the node rejected used to look exactly like one it
                # accepted from in here, so a rejected run was never retried and
                # never billed.
                result = await self._call("crew/message", payload)
                if isinstance(result, dict) and result.get("ok") is False:
                    logger.warning(
                        "run %s settlement was refused: %s",
                        payload.get("runId"),
                        result.get("error"),
                    )
                    return False
                return True
            await self._send(action, payload)
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure is a retry
            logger.warning(
                "could not deliver a %s for run %s: %s",
                kind or "event",
                payload.get("runId"),
                exc or type(exc).__name__,
            )
            return False

    def _highest_sequence(self) -> int:
        """Continue the previous process's numbering rather than restarting it."""
        highest = 0
        for path in self._dir.glob("*.json"):
            head = path.name.split("-", 1)[0]
            if head.isdigit():
                highest = max(highest, int(head))
        return highest
